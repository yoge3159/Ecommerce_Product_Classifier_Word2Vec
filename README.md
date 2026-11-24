# E-Commerce Product Classification using Word2Vec and ANN

## Project Overview
This project is an NLP-based Machine Learning application that automatically predicts the category of an e-commerce product based on its title or description.

It uses Word2Vec for text representation and an Artificial Neural Network (ANN) for classification. The application supports live text input and voice input through Streamlit.

## Technologies Used
- Python
- Natural Language Processing (NLP)
- Word2Vec
- Artificial Neural Network (ANN)
- TensorFlow / Keras
- Gensim
- Scikit-learn
- Streamlit
- Speech Recognition

## Project Workflow
1. Collect and preprocess product text data.
2. Clean and tokenize the text.
3. Generate word embeddings using Word2Vec.
4. Convert product descriptions into numerical vectors.
5. Train an Artificial Neural Network.
6. Predict product categories from new input.
7. Display predictions through a Streamlit web application.

## Features
- Live product text input
- Voice-based product description input
- Automatic product category prediction
- Prediction confidence display
- Interactive Streamlit interface

## Project Structure
```text
Ecommerce_Product_Classifier_Word2Vec/
│
├── artifacts/
│   ├── ann_model.keras
│   ├── word2vec.model
│   └── label_encoder.joblib
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/yoge3159/Ecommerce_Product_Classifier_Word2Vec.git
```

Move into the project folder:

```bash
cd Ecommerce_Product_Classifier_Word2Vec
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Model Details
- Text Representation: Word2Vec
- Classification Algorithm: Artificial Neural Network
- Deep Learning Framework: TensorFlow / Keras
- Interface: Streamlit

## Author
**Yoge**

## Project Links
- GitHub Repository: https://github.com/yoge3159/Ecommerce_Product_Classifier_Word2Vec
- Live Application: Add your Streamlit Cloud URL after deployment.

---
Developed as an end-to-end NLP and Deep Learning project.
</Writing
