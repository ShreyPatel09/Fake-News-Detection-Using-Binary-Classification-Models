# Veracity Vigilance: Fake News Detection System
A comprehensive fake news detection system that uses machine learning and deep learning models to classify news articles as real or fake. The system includes a Jupyter notebook for model development and a Streamlit web application for easy testing and deployment.

## 📋 Features
Multiple ML Models: Implementation of various machine learning approaches including Naive Bayes, Logistic Regression, Random Forest, and Neural Networks (LSTM, Bi-LSTM)

Comprehensive Text Processing: Advanced preprocessing pipeline for optimal text feature extraction

Interactive UI: User-friendly Streamlit interface for testing the models

Performance Metrics: Detailed evaluation using accuracy, precision, recall, and F1 score

Visual Analysis: Visualizations of model performance and text characteristics

## 🔧 Technologies Used
TensorFlow/Keras for deep learning models

scikit-learn for traditional ML models

NLTK for natural language processing

Pandas & NumPy for data manipulation

Matplotlib, Seaborn & Plotly for visualization

Streamlit for the web application

## 📊 Datasets
The system was trained and evaluated on two datasets:

IFND (Indian Fake News Dataset): Contains ~57,000 news articles with "TRUE" or "FAKE" labels

LIAR Dataset: Collection of 12,800+ statements labeled for veracity in different categories

## 🤖 Models and Performance
Model	IFND Accuracy	LIAR Accuracy
Naive Bayes	93.66%	60.02%
Logistic Regression	94.48%	60.64%
Random Forest	95.20%	59.94%
Decision Tree	93.07%	54.17%
KNN	84.99%	55.34%
LSTM	93.81%	55.42%
Bi-LSTM	94.17%	56.12%
The Bi-LSTM model achieved 94.17% accuracy on the IFND dataset with a well-balanced precision and recall.

# 🚀 Installation
-> Clone the repository
git clone https://github.com/yourusername/veracity-vigilance.git
cd veracity-vigilance

-> Install required packages
pip install -r requirements.txt

-> Download necessary NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

## 💻 Usage
-> Model Training (Jupyter Notebook)
Run the Fake_news_detection.ipynb file for training and evaluating the model

**Export model files(.keras extension) for the UI application**

# Web Application
Run the Streamlit app
streamlit run FND_UI.py
-> The web interface will open in your browser

-> Enter news text or use example buttons to analyze content

-> View the detection results and detailed analysis

## 📁 Project Structure
-> Fake_news_detection.ipynb: Jupyter notebook with model development and evaluation

-> FND_UI.py: Streamlit application for model deployment

-> bilstm_model.keras: Saved Bi-LSTM model file

-> requirements.txt: Required Python packages

-> image.png: A flowchart for Model development procedure

-> bilstm_model.keras: Model developed

-> tokenizer.pkl: tokens created
