import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO

#css styling
st.markdown(
    """
    <style>
    /*make sidebar title bigger*/
    section[data-testid="stSidebar"] h1 {
        font-size: 40px;
        font-weight: bold;
    }

    div[data-baseweb="radio"] label {
        font-size: 25px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#load model
@st.cache_resource #only load model once per run
def load_the_model():
    try:
        model = load_model("models/brain_tumour_classifier5.h5")
        return model
    except:
        st.error("Unable to load model. Please try again.")


@st.cache_resource #only load model architecture once per run 
def get_model_architecture():

    #write a line of text to stringio
    def print_to_stringio(text):
        stringio.write(text + "\n")

    stringio = StringIO() #create location to store the model.summary text in memory
    model.summary(print_fn=print_to_stringio) #call print_to_stringio for each line in model.summary(), moving it into stringio
    summary_str = stringio.getvalue() #get text from stringio
    return summary_str 


class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def predict_class(model, img):
    prediction = model.predict(img) #will be an array of probabilities
    predicted_class = np.argmax(prediction, axis=1)[0] #get index of class with highest probability, [0] cuz array
    predicted_label = class_names[predicted_class] 
    st.success(f"Predicted Class: {predicted_label}")
    probs = prediction[0] #get probabilities of each class
    st.info(f"Confidence: {100 * np.max(probs):.2f}%")
    fig, ax = plt.subplots()
    ax.bar(class_names, probs)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Classes")
    ax.set_title("Class Prediction Probabilities")
    st.pyplot(fig)


def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((128, 128))  #change to match your model input
    img = np.array(img) / 255.0  #normalise
    img = np.expand_dims(img, axis=0)  #model expects batch dimension
    return img

st.sidebar.title("Navigation")
#menu
page = st.sidebar.radio(
    "Non Empty Label",
    ["Upload Image For Prediction", "About The Model", "Project Findings", "About The Dev"],
    label_visibility="hidden"
)

model = load_the_model() 

if page == "Upload Image For Prediction":
    st.title("Brain Tumour Classification")
    st.header("Upload an MRI")
    uploaded_image = st.file_uploader("Upload an MRI", type=["jpg", "png"], label_visibility="hidden") #hide label and use header for appearance
    if uploaded_image is not None:
        img = preprocess_image(uploaded_image)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True) #show image on screen
    
    col1, col2 = st.columns([1, 1]) #split the screen
    with col1:
        if st.button("Predict Tumour Class"):
            if uploaded_image is not None:
                with st.spinner("Predicting..."):
                    prediction = predict_class(model, img)
            else:
                st.error("No image uploaded, please try again.")

    with col2:
        if st.button("Clear"):
            st.empty() #clears the screen


elif page == "About The Model":
    st.title("About The Model")

    st.header("Model Architecture")
    summary_str = get_model_architecture()
    st.code(summary_str, language='text')

    st.header("Tech Stack")
    st.subheader("General")
    st.text("Programming language: Python\nIDE: VSCode\nData Source: Kaggle")
    st.subheader("Model Development Frameworks/Libraries")
    st.text("Tensorflow with Keras - machine learning (data pipeline, building the model, training)\n" \
            "Matplotlib - Training visualisation and confusion matrix visualisation\n" \
            "Seaborn - Confusion matrix visualisation\n" \
            "Scikit-Learn - Confusion matrix\n"
            "Numpy - Numerical programming\n")
    st.subheader("Website Development Frameworks/Libraries")
    st.text("Streamlit - Web development\n" \
            "Matplotlib - Visualisation of prediction probabilities\n" \
            "StringIO - Model architecture\n" \
            "PIL - Image preprocessing\n" \
            "Tensorflow with Keras - Model Loading\n" \
            "Numpy - Numerical programming\n")

    st.header("Development")
    st.text("Five different models were developed for this problem. The model used on this website is the 5th of the five, which is the highest performing.")
    st.subheader("The 5 models")
    st.code("""Model No. | Dropout | Data Augmentation | Learning Rate Scheduler | Test Accuracy | Test Loss
Model 1 | No | Yes | No | 0.9420 | 0.2443
Model 2 | Yes | No | No | 0.9382 | 0.2124
Model 3 | No | Yes | No | 0.2288 | 2.9633
Model 4 | Yes | Yes | No | 0.2288 | 2.2432
Model 5 | No | No | Yes | 0.9657 | 0.1469""", language='text')

    st.header("Training Plots")
    st.image("images/trainingplots.png", caption="Training Plots", use_container_width=True)
    st.header("Confusion Matrix")
    st.image("images/confusionmatrix.png", caption="Confusion Matrix", use_container_width=True)
    st.markdown("[Click here for more details on training](https://docs.google.com/document/d/1PlbGaiz6XitTpGNAN_0MSD0Fg6oSjyyNY0aoaInIXlM/edit?usp=sharing)")

elif page == "Project Findings":
    st.title("Findings")
    st.text("The development of these 5 models has shown that various machine learning techniques used to prevent overfitting such as dropout and data augmentation (as tested here) " \
    "can be detrimental to model performance when developing classification models designed to process images where tiny details have a big difference to the class of the image, " \
    "such as medical imaging classifier models like this one. " \
    "The project also gives an idea of the potential for machine learning use in medical settings.")
    st.image("images/brainmris.jpg", use_container_width=True)

elif page == "About The Dev":
    st.title("About The Developer")
    st.text("I'm Henry, a 14 yr old student passionate about computer science and programming. My aim is to apply AI/ML and programming in general to solve meaningful problems. " \
    "This project was a great learning process and has taught me lots about machine learning and computing as a whole. " \
    "I learned the skills needed for this project from 2 courses on kaggle: Intro to Deep Learning and Computer Vision. ")
    st.image("images/binary.png", use_container_width=True)

else:
    st.error("An error has occured.")

with st.sidebar:
    st.image("images/pythonlogo.png", use_container_width=True)
    st.warning("**Disclaimer:** This tool is intended for educational and informational purposes only. " \
    "It is not a substitute for professional medical advice, diagnosis, or treatment. " \
    "Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.")