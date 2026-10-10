"""Application-facing validation, inference, export, and visualization helpers."""

from datetime import datetime
from io import BytesIO
import hashlib
import html
import inspect
import json
import os

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportImage

import create_model as model_module


SUPPORTED_FORMATS = {"PNG", "JPEG"}
MIN_IMAGE_SIDE = 128


def _read_source(source):
    if isinstance(source,(str,os.PathLike)):
        path = os.fspath(source)
        with open(path,"rb") as handle:
            return handle.read(),os.path.basename(path)
    if hasattr(source,"getvalue"):
        return source.getvalue(),getattr(source,"name","uploaded-image")
    raise TypeError("Image source must be a path or uploaded file")


def inspect_image(source):
    """Validate an image and return its RGB pixels, metadata, warnings, and hash."""
    try:
        raw,name = _read_source(source)
        with Image.open(BytesIO(raw)) as opened:
            image_format = (opened.format or "UNKNOWN").upper()
            opened.verify()
        with Image.open(BytesIO(raw)) as opened:
            rgb_image = opened.convert("RGB")
            pixels = np.array(rgb_image)
    except (OSError,UnidentifiedImageError,TypeError,ValueError) as error:
        return {"valid":False,"error":f"Unable to read {getattr(source,'name','image')}: {error}"}

    height,width = pixels.shape[:2]
    if image_format not in SUPPORTED_FORMATS:
        return {"valid":False,"error":f"{name} is {image_format}; upload a PNG or JPEG image."}

    warnings = []
    if min(width,height) < MIN_IMAGE_SIDE:
        warnings.append(f"Low resolution: the shortest side is below {MIN_IMAGE_SIDE} pixels.")
    ratio = max(width,height) / max(1,min(width,height))
    if ratio > 2.5:
        warnings.append("Unusual aspect ratio; confirm this is the intended chest X-ray.")

    return {
        "valid":True,
        "name":name,
        "format":image_format,
        "width":width,
        "height":height,
        "size_bytes":len(raw),
        "sha256":hashlib.sha256(raw).hexdigest(),
        "warnings":warnings,
        "pixels":pixels,
    }


def normalized_pixels(image_info):
    return image_info["pixels"].astype(np.float32) / 255.0


def _image_metadata(image_info):
    return {key:value for key,value in image_info.items() if key != "pixels"}


def generate_report(primary,secondary,model_tokenizer,min_tokens=6):
    """Run inference and return UI-ready report data and model attention."""
    primary_pixels = normalized_pixels(primary)
    secondary_pixels = normalized_pixels(secondary) if secondary else primary_pixels.copy()
    started = datetime.now().astimezone()
    inference_started = datetime.now().timestamp()
    supports_details = "return_details" in inspect.signature(model_module.function1).parameters
    if supports_details:
        results = model_module.function1(
            [primary_pixels],[secondary_pixels],model_tokenizer,
            min_tokens=min_tokens,return_details=True,
        )
        impression,details = results[0]
    else:
        # Compatibility for a Streamlit process that still has the pre-attention
        # decoder module in memory. A process restart enables attention details.
        results = model_module.function1(
            [primary_pixels],[secondary_pixels],model_tokenizer,
            min_tokens=min_tokens,
        )
        impression = results[0]
        details = {"tokens":impression.rstrip(".!?").split(),"attention":np.empty((0,18),dtype=np.float32)}
    elapsed = datetime.now().timestamp()-inference_started
    return {
        "id":started.strftime("%Y%m%d%H%M%S%f"),
        "created_at":started.isoformat(timespec="seconds"),
        "impression":impression,
        "elapsed_seconds":elapsed,
        "view_count":2 if secondary else 1,
        "primary":_image_metadata(primary),
        "secondary":_image_metadata(secondary) if secondary else None,
        "primary_pixels":primary_pixels,
        "secondary_pixels":secondary_pixels,
        "tokens":details["tokens"],
        "attention":details["attention"],
    }


def report_text(report,impression=None):
    impression = impression or report["impression"]
    return (
        "Chest X-ray Generated Impression\n"
        f"Generated: {report['created_at']}\n"
        f"Views supplied: {report['view_count']}\n"
        f"Primary image: {report['primary']['name']}\n"
        f"Secondary image: {report['secondary']['name'] if report['secondary'] else 'Primary view reused'}\n"
        f"Processing time: {report['elapsed_seconds']:.1f} seconds\n\n"
        f"Impression:\n{impression}\n"
    )


def report_json(report,impression=None):
    impression = impression or report["impression"]
    data = {
        "generated_at":report["created_at"],
        "impression":impression,
        "views_supplied":report["view_count"],
        "primary_image":report["primary"]["name"],
        "secondary_image":report["secondary"]["name"] if report["secondary"] else None,
        "processing_seconds":round(report["elapsed_seconds"],2),
    }
    return json.dumps(data,indent=2)


def _pdf_image(image_pixels,max_width=82*mm,max_height=92*mm):
    pixels = np.asarray(image_pixels)
    if np.issubdtype(pixels.dtype,np.floating):
        pixels = np.uint8(np.clip(pixels*255,0,255))
    else:
        pixels = np.uint8(np.clip(pixels,0,255))
    image = Image.fromarray(pixels,"RGB")
    buffer = BytesIO()
    image.save(buffer,format="PNG")
    buffer.seek(0)
    width,height = image.size
    scale = min(max_width/width,max_height/height)
    return ReportImage(buffer,width=width*scale,height=height*scale)


def report_pdf(report,impression=None):
    """Build a PDF with supplied X-rays, report metadata, and impression."""
    impression = impression or report["impression"]
    output = BytesIO()
    document = SimpleDocTemplate(
        output,pagesize=A4,
        rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=16*mm,
        title="Chest X-ray Generated Report",
        author="Chest X-ray Impression Generator",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",parent=styles["Title"],fontSize=19,leading=23,
        textColor=colors.HexColor("#145f72"),alignment=TA_CENTER,spaceAfter=8*mm,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",parent=styles["Heading2"],fontSize=12,leading=15,
        textColor=colors.HexColor("#145f72"),spaceBefore=4*mm,spaceAfter=3*mm,
    ))
    styles.add(ParagraphStyle(
        name="ImpressionBox",parent=styles["BodyText"],fontSize=11,leading=17,
        borderColor=colors.HexColor("#16829a"),borderWidth=1,borderPadding=8,
        backColor=colors.HexColor("#f2f8fa"),spaceAfter=4*mm,
    ))
    styles.add(ParagraphStyle(
        name="ImageCaption",parent=styles["BodyText"],fontSize=8,leading=10,
        alignment=TA_CENTER,textColor=colors.HexColor("#455a64"),
    ))

    story = [Paragraph("Chest X-ray Generated Report",styles["ReportTitle"])]
    summary = [
        ["Report ID",report["id"]],
        ["Generated",report["created_at"]],
        ["Views supplied",str(report["view_count"])],
        ["Generated words",str(len(impression.rstrip(".!?").split()))],
    ]
    summary_table = Table(summary,colWidths=[42*mm,125*mm],hAlign="LEFT")
    summary_table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f7f7f8")),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d6dde0")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.extend([summary_table,Spacer(1,5*mm),Paragraph("Supplied X-rays",styles["SectionTitle"])])

    image_cells = [_pdf_image(report["primary_pixels"])]
    caption_cells = [Paragraph(f"Primary: {html.escape(report['primary']['name'])}",styles["ImageCaption"])]
    if report["secondary"]:
        image_cells.append(_pdf_image(report["secondary_pixels"]))
        caption_cells.append(Paragraph(f"Secondary: {html.escape(report['secondary']['name'])}",styles["ImageCaption"]))
    images_table = Table([image_cells,caption_cells],hAlign="CENTER")
    images_table.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.extend([
        images_table,Spacer(1,4*mm),
        Paragraph("Generated Impression",styles["SectionTitle"]),
        Paragraph(html.escape(impression),styles["ImpressionBox"]),
    ])

    image_rows = [["Image","Format","Dimensions"]]
    image_rows.append([
        Paragraph(html.escape(report["primary"]["name"]),styles["BodyText"]),
        report["primary"]["format"],
        f"{report['primary']['width']} × {report['primary']['height']} px",
    ])
    if report["secondary"]:
        image_rows.append([
            Paragraph(html.escape(report["secondary"]["name"]),styles["BodyText"]),
            report["secondary"]["format"],
            f"{report['secondary']['width']} × {report['secondary']['height']} px",
        ])
    image_table = Table(image_rows,colWidths=[92*mm,28*mm,47*mm],hAlign="LEFT")
    image_table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#145f72")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d6dde0")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.extend([
        Paragraph("Image Information",styles["SectionTitle"]),image_table,
        Paragraph("Model Information",styles["SectionTitle"]),
        Paragraph(
            "DenseNet121/CheXNet visual encoder with spatial attention and a GRU report decoder.",
            styles["BodyText"],
        ),
    ])
    document.build(story)
    return output.getvalue()


def attention_overlay(image_pixels,attention,view_index):
    """Overlay one token's 3x3 model-attention grid on a view."""
    attention = np.asarray(attention,dtype=np.float32).reshape(-1)
    start = 0 if view_index == 0 else 9
    grid = attention[start:start+9]
    if grid.size != 9:
        raise ValueError("Expected 18 attention values for two 3x3 image grids")
    grid = grid.reshape(3,3)
    height,width = image_pixels.shape[:2]
    heat = cv2.resize(grid,(width,height),interpolation=cv2.INTER_CUBIC)
    heat = heat-heat.min()
    if heat.max() > 0:
        heat = heat/heat.max()
    heat_color = cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color,cv2.COLOR_BGR2RGB)
    base = np.uint8(np.clip(image_pixels*255,0,255))
    return np.uint8(cv2.addWeighted(base,0.65,heat_color,0.35,0))
