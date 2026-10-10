import html
import os
import random

import streamlit as st

import inference_service as inference


st.set_page_config(
    page_title="Chest X-ray Impression Generator",
    page_icon="🩻",
    layout="wide",
)

st.markdown("""
<style>
.main .block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.6rem; border-radius: 16px; background: linear-gradient(120deg, #0f2942, #145f72); color: white; margin-bottom: 1.5rem;}
.hero h1 {margin: 0 0 .4rem 0; color: white;}
.hero p {margin: 0; color: #d8edf2;}
.result-card {padding: 1.3rem; border-left: 5px solid #16829a; border-radius: 8px; background: #f2f8fa; color: #102a33; font-size: 1.08rem;}
.meta-card {padding: .8rem 1rem; border-radius: 10px; background: #f7f7f8; margin-bottom: .7rem;}
</style>
<div class="hero">
  <h1>Chest X-ray Impression Generator</h1>
  <p>Generate, inspect, edit, and export a draft impression from one or two chest X-ray views.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return inference.model_module.create_model()


def format_size(size_bytes):
    if size_bytes >= 1024*1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    return f"{size_bytes/1024:.0f} KB"


def show_image_metadata(image_info,label):
    st.markdown(f"**{label} details**")
    st.caption(
        f"{image_info['name']} · {image_info['format']} · "
        f"{image_info['width']} × {image_info['height']} px · {format_size(image_info['size_bytes'])}"
    )
    for warning in image_info["warnings"]:
        st.warning(warning)


def add_report(report):
    st.session_state.reports.insert(0,report)
    st.session_state.reports = st.session_state.reports[:10]
    st.session_state.current_report_id = report["id"]


def generate(primary,secondary,source="upload"):
    with st.spinner("Loading the report-generation model..."):
        model_tokenizer = load_model()
    with st.spinner("Analyzing images and drafting an impression..."):
        report = inference.generate_report(primary,secondary,model_tokenizer,min_tokens=6)
    report["source"] = source
    add_report(report)


def current_report(source=None):
    if not st.session_state.reports:
        return None
    if source:
        return next((report for report in st.session_state.reports if report.get("source") == source), None)
    report_id = st.session_state.current_report_id
    return next((report for report in st.session_state.reports if report["id"] == report_id),None)


def show_report(report,key_prefix="report"):
    st.divider()
    st.markdown("## Generated report")
    badge_col1,badge_col2,badge_col3 = st.columns(3)
    badge_col1.metric("Views supplied",report["view_count"])
    badge_col2.metric("Generation time",f"{report['elapsed_seconds']:.1f}s")
    badge_col3.metric("Words",len(report["impression"].rstrip(".!?").split()))

    st.markdown(
        '<div class="result-card">%s</div>' % html.escape(report["impression"]),
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download PDF report",
        inference.report_pdf(report,report["impression"]),
        file_name=f"xray-report-{report['id']}.pdf",
        mime="application/pdf",
        key=f"{key_prefix}_pdf_{report['id']}",
    )
    with st.expander("Copy-ready impression"):
        st.code(report["impression"],language=None)

    st.markdown("### Model attention")
    st.caption("Shows where the model focused while generating each token; it is not a clinical explanation.")
    if len(report["tokens"]) and len(report["attention"]):
        token_options = [f"{index+1}. {token}" for index,token in enumerate(report["tokens"])]
        selected_label = st.selectbox(
            "Generated token",
            token_options,
            key=f"{key_prefix}_attention_token_{report['id']}",
        )
        token_index = token_options.index(selected_label)
        attention = report["attention"][token_index]
        attention_col1,attention_col2 = st.columns(2)
        primary_overlay = inference.attention_overlay(report["primary_pixels"],attention,0)
        secondary_overlay = inference.attention_overlay(report["secondary_pixels"],attention,1)
        attention_col1.image(primary_overlay,caption="Primary-view model attention",width=360)
        second_caption = "Secondary-view model attention" if report["secondary"] else "Reused-view model attention"
        attention_col2.image(secondary_overlay,caption=second_caption,width=360)
    else:
        st.info("Attention data is unavailable for this report.")


if "reports" not in st.session_state:
    st.session_state.reports = []
if "current_report_id" not in st.session_state:
    st.session_state.current_report_id = None
if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0
if "upload_report_id" not in st.session_state:
    st.session_state.upload_report_id = None


upload_tab,sample_tab,history_tab = st.tabs(["Upload X-rays","Try a sample","Recent reports"])

with upload_tab:
    header_col, reset_col = st.columns([4, 1.2])
    with header_col:
        st.subheader("Add images from the same examination")
        st.caption("Use the frontal view first and the lateral view second when both are available. PNG and JPEG are supported.")
    with reset_col:
        if st.button("New / Reset input", key="reset_upload_btn"):
            st.session_state.uploader_version += 1
            st.session_state.upload_report_id = None
            st.rerun()

    version = st.session_state.uploader_version
    col1,col2 = st.columns(2)
    image_1 = col1.file_uploader("Frontal X-ray (required)",type=["png","jpg","jpeg"],key=f"primary_xray_{version}")
    image_2 = col2.file_uploader("Lateral X-ray (optional)",type=["png","jpg","jpeg"],key=f"secondary_xray_{version}")

    primary = inference.inspect_image(image_1) if image_1 else None
    secondary = inference.inspect_image(image_2) if image_2 else None
    valid_inputs = bool(primary and primary["valid"] and (not secondary or secondary["valid"]))

    if primary:
        if primary["valid"]:
            preview_1,preview_2 = st.columns(2)
            preview_1.image(primary["pixels"],caption="Frontal / primary view",width=360)
            if secondary and secondary["valid"]:
                preview_2.image(secondary["pixels"],caption="Lateral / secondary view",width=360)
            elif not secondary:
                preview_2.info("No second view supplied. The primary image will be reused as the second model input.")
            details_1,details_2 = st.columns(2)
            with details_1:
                show_image_metadata(primary,"Primary image")
            if secondary and secondary["valid"]:
                with details_2:
                    show_image_metadata(secondary,"Secondary image")
            if secondary and secondary["valid"] and primary["sha256"] == secondary["sha256"]:
                st.warning("The two uploaded files are identical. Confirm that you intended to use the same view twice.")
        else:
            st.error(primary["error"])
    if secondary and not secondary["valid"]:
        st.error(secondary["error"])

    if st.button("Generate impression",disabled=not valid_inputs,key="generate_upload"):
        generate(primary,secondary,source="upload")
        st.session_state.upload_report_id = st.session_state.current_report_id

    upload_report = None
    if st.session_state.upload_report_id:
        upload_report = next((r for r in st.session_state.reports if r["id"] == st.session_state.upload_report_id), None)
    if upload_report:
        show_report(upload_report,key_prefix="upload")

with sample_tab:
    st.write("Run the model with one of the chest X-ray examples included in this repository.")
    if st.button("Generate from a random sample",key="generate_sample"):
        sample_folders = [entry.path for entry in os.scandir("./test_images") if entry.is_dir()]
        usable = []
        for folder in sample_folders:
            images = sorted(
                os.path.join(folder,name) for name in os.listdir(folder)
                if name.lower().endswith((".png",".jpg",".jpeg"))
            )
            if images:
                usable.append(images)
        if not usable:
            st.error("No sample images were found.")
        else:
            selected = usable[random.randrange(len(usable))]
            sample_primary = inference.inspect_image(selected[0])
            sample_secondary = inference.inspect_image(selected[1]) if len(selected) > 1 else None
            generate(sample_primary,sample_secondary,source="sample")

    sample_report = current_report(source="sample")
    if sample_report:
        preview_1,preview_2 = st.columns(2)
        preview_1.image(sample_report["primary_pixels"],caption="Frontal / primary view",width=360)
        if sample_report.get("secondary_pixels") is not None and sample_report.get("secondary"):
            preview_2.image(sample_report["secondary_pixels"],caption="Lateral / secondary view",width=360)
        show_report(sample_report,key_prefix="sample")

with history_tab:
    if not st.session_state.reports:
        st.info("Generated reports from this browser session will appear here.")
    else:
        heading_col,clear_col = st.columns([4,1])
        heading_col.subheader(f"Recent reports ({len(st.session_state.reports)})")
        if clear_col.button("Clear history"):
            st.session_state.reports = []
            st.session_state.current_report_id = None
            st.session_state.selected_history_id = None
            st.rerun()
        for report in st.session_state.reports:
            row_image,row_text,row_action = st.columns([1,3,1])
            row_image.image(report["primary_pixels"],width=110)
            row_text.markdown(f"**{report['impression']}**")
            row_text.caption(f"{report['created_at']} · {report['view_count']} view(s) · {report['elapsed_seconds']:.1f}s")
            if row_action.button("Open",key=f"open_{report['id']}"):
                st.session_state.selected_history_id = report["id"]
                st.rerun()
            st.divider()

        if st.session_state.selected_history_id:
            selected_history_report = next((r for r in st.session_state.reports if r["id"] == st.session_state.selected_history_id), None)
            if selected_history_report:
                st.subheader("Selected Report Details")
                show_report(selected_history_report,key_prefix="history")
