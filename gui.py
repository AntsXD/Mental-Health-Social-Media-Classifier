import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # Added this
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Define the AdvancedTextPreprocessor class
class AdvancedTextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Keep negations and important words for mental health
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never', 'none'}
    
    def clean_text(self, text):
        """Basic cleaning"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
        text = re.sub(r'@\w+', '', text)  # Mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Hashtags (keep word)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Extra spaces
        return text
    
    def tokenize_text(self, text):
        """Tokenization using NLTK"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatization"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens):
        """Stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text, use_stemming=False):
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_text(cleaned)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        else:
            tokens = self.lemmatize_tokens(tokens)
        return tokens, ' '.join(tokens)
    
    def transform(self, texts):
        """Transform method for compatibility with sklearn pipelines"""
        if isinstance(texts, str):
            texts = [texts]
        return [self.preprocess(text)[1] for text in texts]

# Page configuration
st.set_page_config(
    page_title="Mental Health Detection",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Mental Health Detection System")
st.markdown("Analyze text using ML and Deep Learning models to detect mental health concerns")

# Load models and preprocessors
@st.cache_resource
def load_models_and_preprocessors():
    """Load all models and preprocessing tools"""
    try:
        # Load classical ML model and its preprocessor
        classical_model = joblib.load('best_classical_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        # Load LSTM model and its tokenizer
        lstm_model = keras.models.load_model('mental_health_lstm_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        st.sidebar.success("✅ All models loaded successfully!")
        
        return {
            'classical_model': classical_model,
            'tfidf_vectorizer': tfidf_vectorizer,
            'preprocessor': preprocessor,
            'lstm_model': lstm_model,
            'tokenizer': tokenizer
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load everything at startup
models_dict = load_models_and_preprocessors()

# Sidebar for model selection
st.sidebar.header("Model Selection")

# Model options
model_options = {
    "Classical ML Model": "classical",
    "LSTM Deep Learning Model": "lstm"
}

selected_model_name = st.sidebar.selectbox(
    "Select Model:",
    list(model_options.keys())
)

selected_model_type = model_options[selected_model_name]

# Display model info
st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Model:** {selected_model_name}")

# Add model description
if selected_model_type == "classical":
    st.sidebar.markdown("""
    **Classical ML Model**
    - Uses TF-IDF vectorization
    - Traditional machine learning algorithm
    - Fast inference
    """)
else:
    st.sidebar.markdown("""
    **LSTM Model**
    - Deep learning architecture
    - Captures sequential patterns
    - Better for complex contexts
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Text")
    
    # Initialize session state if not exists
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    text_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.text_input,
        height=200,
        placeholder="Type or paste text here for mental health analysis...",
        key="text_area_input"
    )
    
    # Update session state when text changes
    st.session_state.text_input = text_input
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        predict_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)



# Prediction functions
def predict_classical(text, models_dict):
    """Predict using classical ML model"""
    try:
        # Preprocess the text
        if models_dict['preprocessor'] is not None:
            processed_text = models_dict['preprocessor'].transform([text])
        else:
            processed_text = [text]
        
        # Vectorize using TF-IDF
        text_vectorized = models_dict['tfidf_vectorizer'].transform(processed_text)
        
        # Get prediction and probabilities
        prediction_raw = models_dict['classical_model'].predict(text_vectorized)[0]
        
        # Map 0/1 to readable labels (FLIPPED: 0=Concern, 1=No Concern)
        label_map = {1: "Mental Health Concern", 0: "No Mental Health Concern"}
        
        # Try to get probability scores if available
        if hasattr(models_dict['classical_model'], 'predict_proba'):
            probabilities = models_dict['classical_model'].predict_proba(text_vectorized)[0]
            classes = [label_map.get(cls, str(cls)) for cls in models_dict['classical_model'].classes_]
            prediction = label_map.get(prediction_raw, str(prediction_raw))
        else:
            # If no predict_proba, just return binary
            probabilities = [1.0]
            prediction = label_map.get(prediction_raw, str(prediction_raw))
            classes = [prediction]
        
        return prediction, probabilities, classes
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def predict_lstm(text, models_dict):
    """Predict using LSTM model"""
    try:
        # Tokenize and pad the text
        tokenizer = models_dict['tokenizer']
        sequence = tokenizer.texts_to_sequences([text])
        
        # Pad sequence (adjust max_length based on your training)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        max_length = 100  # Adjust this to match your training
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        # Get prediction
        probabilities = models_dict['lstm_model'].predict(padded_sequence, verbose=0)[0]
        
        # Map 0/1 to readable labels (FLIPPED: 0=Concern, 1=No Concern)
        label_map = {0: "Mental Health Concern", 1: "No Mental Health Concern"}
        
        # Assuming binary or multi-class classification
        if len(probabilities.shape) == 0 or probabilities.shape[0] == 1:
            # Binary classification
            prob_value = float(probabilities[0]) if len(probabilities.shape) > 0 else float(probabilities)
            prediction = label_map[0] if prob_value > 0.5 else label_map[1]
            classes = [label_map[0], label_map[1]]
            probabilities = [prob_value, 1 - prob_value]
        else:
            # Multi-class classification
            # Update with actual class names if you have them
            classes = [label_map.get(i, f"Class {i}") for i in range(len(probabilities))]
            prediction_idx = np.argmax(probabilities)
            prediction = classes[prediction_idx]
        
        return prediction, probabilities, classes
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# Handle clear button
if clear_button:
    st.session_state.text_input = ""
    st.rerun()

# Handle predictions
if predict_button:
    if not text_input.strip():
        st.error("⚠️ Please enter some text to analyze")
    elif models_dict is None:
        st.error("⚠️ Models failed to load. Please check your model files.")
    else:
        with st.spinner(f"Running {selected_model_name}..."):
            # Make prediction based on selected model
            if selected_model_type == "classical":
                prediction, probabilities, classes = predict_classical(text_input, models_dict)
            else:
                prediction, probabilities, classes = predict_lstm(text_input, models_dict)
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main prediction
                max_prob = max(probabilities)
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Predicted Class", prediction)
                with col_res2:
                    st.metric("Confidence", f"{max_prob*100:.2f}%")
                
                # Confidence level indicator
                if max_prob > 0.8:
                    st.success("✅ High confidence prediction")
                elif max_prob > 0.6:
                    st.info("ℹ️ Moderate confidence prediction")
                else:
                    st.warning("⚠️ Low confidence prediction - results may be uncertain")
                
                # All scores
                st.subheader("All Class Probabilities")
                
                # Create DataFrame for visualization
                scores_df = pd.DataFrame({
                    'Class': classes,
                    'Probability': [f"{p*100:.2f}%" for p in probabilities],
                    'Score': probabilities
                }).sort_values('Score', ascending=False)
                
                # Display as bar chart
                st.bar_chart(scores_df.set_index('Class')['Score'])
                
                # Display as table
                st.dataframe(
                    scores_df[['Class', 'Probability']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Disclaimer
                st.markdown("---")
                st.warning("⚠️ **Disclaimer:** This is an automated analysis tool and should not replace professional mental health assessment. If you're experiencing mental health concerns, please consult a qualified healthcare professional.")

# Footer
st.markdown("---")
st.markdown("""
### About the Models
- **Classical ML Model:** Uses TF-IDF features with traditional machine learning
- **LSTM Model:** Deep learning model that captures sequential patterns in text

### Files Used
- `best_classical_model.pkl` - Trained ML classifier
- `mental_health_lstm_model.h5` - Trained LSTM model
- `tfidf_vectorizer.pkl` - Text-to-features converter (for ML model)
- `tokenizer.pkl` - Text-to-sequences converter (for LSTM)
- `preprocessor.pkl` - Text preprocessing pipeline
""")