import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import time
from collections import Counter
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Download NLTK resources if not already downloaded
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance UI appearance
st.markdown(
    """
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 28px;
        font-weight: bold;
        color: #2563EB;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .description {
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 25px;
    }
    .highlight {
        background-color: #2C2C33;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .fake-result {
        background-color: red;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #EF4444;
        text-align: center;
    }
    .real-result {
        background-color: green;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #10B981;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        background-color: #2C2C33;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        transform: translateY(-2px);
    }
    .clear-button>button {
        background-color: #9CA3AF;
    }
    .clear-button>button:hover {
        background-color: #4B5563;
    }
    .example-button>button {
        background-color: #8B5CF6;
    }
    .example-button>button:hover {
        background-color: #6D28D9;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Sidebar with enhanced styling
st.sidebar.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
st.sidebar.info(
    "This application uses deep learning to detect fake news. "
    "It was trained on the IFND (Indian Fake News Dataset) using "
    "a Bi-LSTM architecture that achieved 95.34% accuracy."
)


st.sidebar.markdown(
    '<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True
)
metrics = {"Accuracy": 94.17, "Precision": 93.45, "Recall": 93.28, "F1 Score": 93.33}

# Create a metrics visualization
metrics_fig = go.Figure(
    go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker_color=["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"],
    )
)
metrics_fig.update_layout(
    yaxis=dict(range=[80, 100]),
    height=250,
    margin=dict(l=20, r=20, t=20, b=40),
)
st.sidebar.plotly_chart(metrics_fig, use_container_width=True)

# App header with enhanced styling
st.markdown(
    '<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True
)
st.markdown(
    "<div class=\"description highlight\">This advanced application helps you identify potential fake news articles using a Bidirectional LSTM neural network. Simply paste the news text in the box below and click 'Analyze' to get an instant prediction with confidence score.</div>",
    unsafe_allow_html=True,
)


# Cache the model loading to prevent reloading on each interaction
@st.cache_resource
def load_dl_model():
    try:
        model = load_model("bilstm_model.keras")
        with open("tokenizer.pkl", "rb") as file:
            tokenizer = pickle.load(file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading DL model: {e}")
        return None, None


# Load preprocessing objects
@st.cache_resource
def load_preprocessing_objects():
    try:
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        return stemmer, stop_words
    except Exception as e:
        st.error(f"Error loading preprocessing objects: {e}")
        return None, None


# Text preprocessing function
def preprocess_text(text, stemmer, stop_words):
    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a single string
    text = " ".join(tokens)

    return text


# Function to predict using DL model
def predict_dl(text, model, tokenizer, stemmer, stop_words, max_len=50):
    # Preprocess the text
    processed_text = preprocess_text(text, stemmer, stop_words)

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Make prediction
    pred_proba = model.predict(padded_sequence)[0][0]
    pred_class = 1 if pred_proba > 0.5 else 0

    return pred_class, pred_proba, processed_text


# Sample news examples
fake_news_example = """BREAKING: Scientists confirm COVID-19 vaccine contains microchip to track people. Bill Gates and Elon Musk are behind this conspiracy to track the world population. Government is hiding the truth! Share before this gets deleted!"""

real_news_example = """The World Health Organization announced today that global COVID-19 cases have declined by 15% over the past month, according to data collected from 150 countries. Health officials attribute this drop to increased vaccination rates and continued public health measures."""

# Enhanced user input section with tabs
st.markdown('<div class="sub-header">Analyze News Text</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Manual Entry", "Examples"])

with tab1:
    news_text = st.text_area(
        "Enter the headline or complete news article text below:",
        height=200,
        placeholder="Paste news text here to check if it's real or fake...",
    )

    # Buttons row with better styling
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        clear_button = st.button("Clear", key="clear", help="Clear the text area")
        st.markdown("<style>.clear-button {}</style>", unsafe_allow_html=True)

    with col2:
        analyze_button = st.button(
            "Analyze", type="primary", key="analyze", help="Analyze the text"
        )

    # Word count indicator
    if news_text:
        word_count = len(news_text.split())
        st.caption(
            f"Word count: {word_count} | Optimal analysis requires at least 20 words"
        )
        # Progress bar for word count
        if word_count < 20:
            st.progress(min(word_count / 20, 1.0))
            if word_count < 10:
                st.info(
                    "For best results, please provide more text (at least 20 words recommended)."
                )

with tab2:
    st.markdown("### Example News Texts")
    st.info("Click one of the examples below to see how the detector works")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Fake News Example", key="fake_example"):
            st.session_state.news_text = fake_news_example
            st.rerun()

    with col2:
        if st.button("Load Real News Example", key="real_example"):
            st.session_state.news_text = real_news_example
            st.rerun()

# Store text in session state to persist between interactions
if "news_text" in st.session_state:
    news_text = st.session_state.news_text

# Clear button functionality
if clear_button:
    st.session_state.news_text = ""
    st.rerun()


# Analysis section
if analyze_button or (
    "analyze_pressed" in st.session_state and st.session_state.analyze_pressed
):
    st.session_state.analyze_pressed = True

    if not news_text or news_text.isspace():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Store the text in session state
        st.session_state.news_text = news_text

        # Show an engaging spinner while processing
        with st.spinner("🔍 AI model analyzing text..."):
            # Add a slight delay to show the spinner (optional)
            time.sleep(0.5)

            # Load models and preprocessing objects
            dl_model, tokenizer = load_dl_model()
            stemmer, stop_words = load_preprocessing_objects()

            if dl_model and tokenizer:
                # Make DL prediction
                dl_class, dl_proba, processed_text = predict_dl(
                    news_text, dl_model, tokenizer, stemmer, stop_words
                )

                # Display results with animation effect
                st.markdown(
                    '<div class="sub-header">Analysis Result</div>',
                    unsafe_allow_html=True,
                )

                # Determine prediction label and color
                prediction = "REAL" if dl_class == 1 else "FAKE"
                probability = dl_proba if dl_class == 1 else 1 - dl_proba

                # Display the result in a highlighted box
                result_class = "real-result" if dl_class == 1 else "fake-result"
                st.markdown(
                    f'<div class="{result_class}"><h1>{prediction} NEWS</h1><h3>Confidence: {probability:.2%}</h3></div>',
                    unsafe_allow_html=True,
                )

                # Create two columns for visualizations
                col1, col2 = st.columns(2)

                with col1:
                    # Create a gauge chart with Plotly for visualization
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Confidence Score", "font": {"size": 24}},
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickwidth": 1,
                                    "tickcolor": "darkblue",
                                },
                                "bar": {
                                    "color": "#2563EB" if dl_class == 1 else "#DC2626"
                                },
                                "bgcolor": "white",
                                "borderwidth": 2,
                                "bordercolor": "gray",
                                "steps": [
                                    {"range": [0, 50], "color": "#FEE2E2"},
                                    {"range": [50, 100], "color": "#DCFCE7"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Create a pie chart for classification breakdown
                    labels = ["Real", "Fake"]
                    values = (
                        [probability, 1 - probability]
                        if dl_class == 1
                        else [1 - probability, probability]
                    )
                    colors = ["#10B981", "#DC2626"]

                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=labels,
                                values=values,
                                hole=0.4,
                                marker_colors=colors,
                            )
                        ]
                    )
                    fig.update_layout(
                        title_text="Classification Breakdown",
                        height=300,
                        annotations=[
                            dict(
                                text=f"{prediction}",
                                x=0.5,
                                y=0.5,
                                font_size=20,
                                showarrow=False,
                            )
                        ],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Add tabs for detailed analysis
                details_tab1, details_tab2 = st.tabs(["Processed Text", "Key Features"])

                with details_tab1:
                    st.subheader("Text After Preprocessing")
                    st.code(processed_text)
                    st.caption(
                        "The text above shows how your content appears to our AI model after removing stopwords, stemming, and other preprocessing steps."
                    )

                with details_tab2:
                    st.subheader("Term Frequency Analysis")

                    try:
                        # Load the IFND.csv dataset
                        # You'll need to adjust this path to where your dataset is stored
                        df = pd.read_csv("IFND_cleaned.csv", encoding="ISO-8859-1")

                        # Combine all text from the 'Statement' column
                        all_text = " ".join(
                            df["Statement"].dropna().astype(str).tolist()
                        )

                        # Simple preprocessing: lowercase and split
                        all_words = all_text.lower().split()

                        # Count word frequencies
                        word_freq = Counter(all_words)

                        # Convert to DataFrame for plotting
                        freq_df = pd.DataFrame(
                            word_freq.items(), columns=["Word", "Frequency"]
                        )

                        # Sort by frequency descending
                        freq_df = freq_df.sort_values(
                            by="Frequency", ascending=False
                        ).head(
                            20
                        )  # Top 20 words

                        # Plot bar chart using plotly
                        fig = px.bar(
                            freq_df,
                            x="Frequency",
                            y="Word",
                            orientation="h",
                            color="Frequency",
                            color_continuous_scale="Viridis",
                            title="Top 20 Term Frequencies in IFND Dataset",
                        )
                        fig.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            height=500,
                            margin=dict(l=20, r=20, t=50, b=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show the top words as a table as well
                        st.subheader("Top 10 Most Frequent Terms")
                        st.table(freq_df.head(10))

                    except Exception as e:
                        st.error(f"Error analyzing term frequencies: {e}")
                        st.info(
                            "Make sure the IFND.csv file is available in the application directory."
                        )


# Add informative expandable sections
with st.expander("How does the detection work?"):
    col1, col2 = st.columns([1, 2])

    with col1:
        # Placeholder for a diagram
        st.image("image.png", caption="Fake News Detection Process")

    with col2:
        st.markdown(
            """
        ### The detection process involves several steps:
        
        1. **Text Preprocessing**: Converting to lowercase, removing URLs, special characters, and numbers, removing common stopwords, and stemming words to their root form.
        
        2. **Feature Extraction**: The text is converted into numerical data that the neural network can understand.
        
        3. **Bidirectional LSTM Analysis**: The model processes the text sequence in both forward and backward directions to capture context effectively.
        
        4. **Classification**: The model outputs a probability score indicating whether the news is likely real or fake.
        
        This approach achieved 95.34% accuracy on the IFND dataset, significantly outperforming traditional methods.
        """
        )

with st.expander("About the Bi-LSTM model"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### Technical details:

        - **Architecture**: Bidirectional LSTM neural network
        - **Embedding**: One-hot encoding with dimension 40
        - **Hidden Layers**: 100 LSTM units with bidirectional processing
        - **Regularization**: Dropout (0.3) to prevent overfitting
        - **Training Dataset**: IFND (Indian Fake News Dataset)
        - **Validation Split**: 33% of data used for testing
        - **Epochs**: Trained for 10 epochs with early stopping
        - **Optimizer**: Adam with learning rate 0.001
        - **Loss Function**: Binary cross-entropy
        """
        )

    with col2:

        epochs = list(range(1, 11))
        acc = [
            0.8573,
            0.9587,
            0.9640,
            0.9699,
            0.9751,
            0.9773,
            0.9802,
            0.9841,
            0.9859,
            0.9886,
        ]
        val_acc = [
            0.9459,
            0.9472,
            0.9475,
            0.9475,
            0.9464,
            0.9453,
            0.9510,
            0.9483,
            0.9464,
            0.9405,
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=epochs, y=acc, mode="lines+markers", name="Training Accuracy")
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=val_acc, mode="lines+markers", name="Validation Accuracy"
            )
        )
        fig.update_layout(
            title="Model Training History", xaxis_title="Epochs", yaxis_title="Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)

# Enhanced footer
st.markdown(
    """
<div class="footer">
    <h3>Fake News Detection - Powered by Bidirectional LSTM</h3>
    <p>Model Accuracy: 94.17%</p>
</div>
""",
    unsafe_allow_html=True,
)
