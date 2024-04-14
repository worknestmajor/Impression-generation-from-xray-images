import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm
import sys

logo_image = Image.open("C:\\Users\\patil\\Documents\\abc\\Summarized-Medical-Report-Generation-on-Chest-X-Rays-main\\Summarized-Medical-Report-Generation-on-Chest-X-Rays-main\\test_images\\logog-removebg-preview.png")
col1, col2 = st.columns([1, 5])  # Divide the row into two columns, with the logo taking up 1/6 of the width
with col1:
    
    st.image(logo_image, width=100)


def load_background_image():
    image_path = "background.jpg"  # Path to your background image file
    image_style = f"""
        <style>
            .stApp {{
                background-image: url("https://img.freepik.com/free-photo/white-brick-wall_1194-7432.jpg?w=826&t=st=1711220371~exp=1711220971~hmac=54825c85d68624b5fc91e7e0534cec42286e517f9bd44e9d9c7da7232ad4f14b");
                background-size: cover;
            }}
        </style>
    """
    st.markdown(image_style, unsafe_allow_html=True)

# Load background image
load_background_image()
with col2:
    st.title("Impression Generation using X Ray Images")
#st.markdown("[Github](https://github.com/fastio19/Summarized-Medical-Report-Generation-on-Chest-X-Rays)")
#st.markdown("\nThis app will generate impression part of an X-ray report.\nYou can upload 2 X-rays that are front view and side view of chest of the same individual.")
#st.markdown("The 2nd X-ray is optional.")


col1,col2 = st.columns(2)
image_1 = col1.file_uploader("X-ray 1",type=['png','jpg','jpeg'])
image_2 = None
if image_1:
    image_2 = col2.file_uploader("X-ray 2 (optional)",type=['png','jpg','jpeg'])

col1,col2 = st.columns(2)
predict_button = col1.button('Predict on uploaded files')
test_data = col2.button('Predict on sample data')

@st.cache(allow_output_mutation=True)
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict(image_1,image_2,model_tokenizer,predict_button = predict_button):
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
            image_1 = np.array(image_1)/255
            if image_2 is None:
                image_2 = image_1
            else:
                image_2 = Image.open(image_2).convert("RGB") #converting to 3 channels
                image_2 = np.array(image_2)/255
            st.image([image_1,image_2],width=300)
            caption = cm.function1([image_1],[image_2],model_tokenizer)
            with st.markdown('<div style="background-color:#FFFFFF; padding: 10px; border-radius: 5px;">' + "Impression:" + '</div>', unsafe_allow_html=True):
                pass
            with st.markdown('<div style="background-color:#FFFFFF; padding: 10px; border-radius: 5px;">' + caption[0] + '</div>', unsafe_allow_html=True):
                pass
            time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
            with st.markdown('<div style="background-color:#FFFFFF; padding: 10px; border-radius: 5px;">' + time_taken + '</div>', unsafe_allow_html=True):
                pass
            del image_1,image_2
        else:
            st.markdown("## Upload an Image")

def predict_sample(model_tokenizer,folder = './test_images'):
    no_files = len(os.listdir(folder))
    file = np.random.randint(1,no_files)
    file_path = os.path.join(folder,str(file))
    if len(os.listdir(file_path))==2:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = os.path.join(file_path,os.listdir(file_path)[1])
        print(file_path)
    else:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = image_1
    predict(image_1,image_2,model_tokenizer,True)
    

model_tokenizer = create_model()


if test_data:
    predict_sample(model_tokenizer)
else:
    predict(image_1,image_2,model_tokenizer)
