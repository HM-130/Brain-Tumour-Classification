# Brain Tumour MRI Classification using Deep Learning

This project is a convolutional neural network (CNN) designed to classify brain MRI scans into different categories of brain tumours. The model is deployed as an interactive Streamlit web application that allows users to upload MRI images and receive real-time predictions with visualised confidence scores.

---

## Project Overview

The goal of this project was to explore how machine learning can be applied to medical imaging, and to better understand how computational models relate to underlying biological structures represented in MRI data.

Rather than treating this as a purely image classification problem, I aimed to investigate how differences in brain tissue structure and tumour biology are reflected in imaging data and learned by deep learning models.

---

## Features

- CNN-based image classification model built using TensorFlow / Keras  
- Classification of MRI scans into:
  - Glioma  
  - Meningioma  
  - Pituitary tumour  
  - No tumour  
- Interactive web app built with Streamlit  
- Real-time prediction with probability outputs  
- Visualisation of model confidence scores  
- Confusion matrix analysis of model performance  
- Training vs validation accuracy/loss tracking  
- Model comparison across multiple experiments  

---

## Key Insights

- MRI classification is not purely a pattern recognition task, but an indirect representation of biological processes such as tissue density, vascular structure, and tumour heterogeneity.  
- Model performance must be interpreted alongside biological plausibility, not just accuracy metrics.  
- Techniques used to reduce overfitting (e.g. dropout and data augmentation) can have trade-offs when fine-grained biological features are important for classification.  
- Understanding model errors (e.g. via confusion matrices) provides insight into similarities between tumour types in imaging space.

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Pillow (PIL)  
- Streamlit  

---

## Model Development

Five different models were trained and evaluated during development. The final deployed model was selected based on highest validation performance.

Experiments included:
- dropout regularisation  
- data augmentation  
- learning rate scheduling  

The final model achieved the best balance between accuracy and generalisation.

---

## App Interface

The Streamlit application allows users to:
- Upload MRI images  
- View predicted tumour class  
- See prediction confidence levels  
- Visualise class probability distribution  

---

## How to Run Locally

(bash)
git clone https://github.com/your-username/brain-tumour-classifier.git
cd brain-tumour-classifier
pip install -r requirements.txt
streamlit run app.py

## Learning Outcomes

This project helped me develop a stronger understanding of:

- convolutional neural networks and image classification  
- the challenges of applying machine learning to medical data  
- the relationship between biological systems and computational models  
- evaluating model performance beyond simple accuracy metrics  

---

## Disclaimer

This project is intended for educational purposes only and is not a diagnostic tool. It should not be used for medical decision-making.

---

## Author

Developed by Me (wow)  
Secondary school student interested in the intersection of biology and machine learning.

---

## Acknowledgements

- Kaggle datasets and learning resources  
- TensorFlow and Streamlit documentation  
