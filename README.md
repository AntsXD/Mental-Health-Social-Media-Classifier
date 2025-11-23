# Mental Health Social Media Classifier
## NLP Course Project - Fall 2025

### 📋 Project Overview
This project implements a comprehensive NLP system for detecting mental health concerns in social media posts. It demonstrates advanced text preprocessing, feature extraction, classical machine learning, and deep learning techniques.

---

## 🎯 Problem Statement
Mental health issues affect millions globally, and social media provides early signals of distress. This project builds an AI-powered classifier to automatically detect mental health concerns in social media posts for early intervention and support.

---

## 📊 Datasets

The project uses two Kaggle datasets:

1. **Depression: Reddit Dataset (Cleaned)**
   - URL: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
   - Download and place as `Mental_Health_Dataset.csv`

2. **Twitter Depression Dataset**
   - URL: https://www.kaggle.com/datasets/hyunkic/twitter-depression-dataset
   - Download and place as `twitter_depression.csv`

**Note:** If datasets are not available, the code will generate sample data for demonstration.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 4GB RAM
- GPU (optional, for faster deep learning training)

### Step 1: Clone or Download the Project
```bash
# Create project directory
mkdir mental_health_nlp
cd mental_health_nlp

# Place the main code file here
# mental_health_classifier.py
```

### Step 2: Install Required Libraries
```bash
# Install core dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Install NLP libraries
pip install nltk spacy textblob

# Install deep learning
pip install tensorflow

# Install word embeddings
pip install gensim

# Download spaCy model
python -m spacy download en_core_web_sm

# Optional: For demo interface
pip install streamlit gradio
```

### Step 3: Download NLTK Data
The code automatically downloads required NLTK data, but you can manually download:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
```

---

## 📁 Project Structure

```
mental_health_nlp/
│
├── mental_health_classifier.py    # Main project code
├── Mental_Health_Dataset.csv      # Dataset 1 (download from Kaggle)
├── twitter_depression.csv         # Dataset 2 (download from Kaggle)
│
├── README.md                       # This file
├── CODE_EXPLANATION.md            # Detailed code explanation
│
├── Output Files (Generated):
│   ├── frequency_analysis.png
│   ├── sentiment_analysis.png
│   ├── topic_clustering.png
│   ├── training_history.png
│   ├── comprehensive_evaluation.png
│   ├── feature_importance.png
│   ├── error_analysis.png
│   │
│   ├── mental_health_lstm_model.h5
│   ├── preprocessor.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer.pkl
│   ├── best_classical_model.pkl
│   └── word2vec_model.bin
```

---

## ▶️ How to Run

### Option 1: Run Complete Pipeline
```bash
python mental_health_classifier.py
```

This will execute all steps:
1. Load and explore data
2. Preprocess text
3. Perform linguistic analysis
4. Analyze word frequencies
5. Sentiment analysis
6. Feature extraction (TF-IDF, Word2Vec)
7. Topic clustering (LDA, K-Means)
8. Train classical models (Naive Bayes, Logistic Regression, Random Forest)
9. Train deep learning model (Bidirectional LSTM with Attention)
10. Comprehensive evaluation
11. Generate visualizations
12. Save models

### Option 2 : To use streamlit please use:

```bash
streamlit run gui.py 
```

Make sure to have the models and all pipelines in the same directory


---

## 🎨 Generated Outputs

### Visualizations (PNG files):
1. **frequency_analysis.png** - Word frequency distributions
2. **sentiment_analysis.png** - Sentiment score distributions
3. **topic_clustering.png** - LDA topics and K-Means clusters
4. **training_history.png** - Deep learning training metrics
5. **comprehensive_evaluation.png** - Model comparison and ROC curves
6. **feature_importance.png** - Most important features
7. **error_analysis.png** - Misclassification patterns

### Saved Models:
1. **mental_health_lstm_model.h5** - Trained LSTM model
2. **preprocessor.pkl** - Text preprocessor
3. **tfidf_vectorizer.pkl** - TF-IDF vectorizer
4. **tokenizer.pkl** - Keras tokenizer
5. **best_classical_model.pkl** - Best classical ML model
6. **word2vec_model.bin** - Word2Vec embeddings

---

## 🧪 Using Trained Models for Predictions

```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models
dl_model = load_model('mental_health_lstm_model.h5')
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict function
def predict_text(text):
    # Preprocess
    tokens, cleaned = preprocessor.preprocess(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    
    # Predict
    proba = dl_model.predict(padded)[0][0]
    prediction = "Mental Health Concern" if proba > 0.5 else "Normal"
    
    print(f"Text: {text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {proba:.2%}")
    return prediction, proba

# Example usage
predict_text("I feel so depressed and anxious today")
predict_text("Having a wonderful day with friends!")
```

---

##  Troubleshooting

### Issue: NLTK data not found
```bash
# Solution: Manually download
python -c "import nltk; nltk.download('all')"
```

### Issue: spaCy model not found
```bash
# Solution: Download model
python -m spacy download en_core_web_sm
```

### Issue: TensorFlow/CUDA errors
```bash
# Solution: Install CPU version
pip install tensorflow-cpu
```

### Issue: Memory errors
```python
# Solution: Reduce batch size or max features
# In the code, modify:
tfidf_vectorizer = TfidfVectorizer(max_features=500)  # Instead of 1000
# Also reduce LSTM batch size to 16
```

### Issue: Datasets not loading
- Ensure CSV files are in the same directory as the script
- Check file names match exactly: `Mental_Health_Dataset.csv` and `twitter_depression.csv`
- The code will use sample data if files are not found

---



## 🎓 NLP Techniques Demonstrated

### 1. **Text Preprocessing**
   - Tokenization (NLTK, spaCy)
   - Lemmatization & Stemming
   - Stopword removal
   - Text cleaning (URLs, mentions, special characters)

### 2. **Linguistic Analysis**
   - POS (Part-of-Speech) tagging
   - Dependency parsing
   - Noun phrase chunking
   - Named entity recognition

### 3. **Feature Extraction**
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Word2Vec embeddings
   - N-gram features
   - Sentiment features

### 4. **Sentiment Analysis**
   - VADER sentiment scoring
   - TextBlob polarity and subjectivity

### 5. **Topic Modeling**
   - LDA (Latent Dirichlet Allocation)
   - K-Means clustering

### 6. **Classification Models**
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - **Bidirectional LSTM with Attention** (Deep Learning)

### 7. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curves
   - Confusion matrices
   - Cross-validation

### 8. **Visualization**
   - Word frequency analysis
   - Sentiment distributions
   - Model performance comparison
   - Training history plots
   - Error analysis

---



### Key Points to Highlight:
- Comprehensive NLP pipeline (13+ techniques)
- Deep learning model built from scratch
- High accuracy (>90% with LSTM)
- Interpretable results (feature importance, error analysis)
- Production-ready (saved models for deployment)

---

##  Future Enhancements

1. **Transformer Models**: BERT, RoBERTa fine-tuning
2. **Multi-class Classification**: Detect specific conditions (depression, anxiety, PTSD)
3. **Real-time Deployment**: Web app with Streamlit/Flask
4. **Explainability**: LIME/SHAP for model interpretability
5. **Multilingual Support**: Extend to other languages
6. **Temporal Analysis**: Track mental health trends over time

---

## References

### Datasets:
Datasets have been modified a bit

- InFamousCoder (2022). Depression: Reddit Dataset (Cleaned). Kaggle.
- Hyunkic. (2024). Twitter Depression Dataset. Kaggle.

### Libraries:
- NLTK: Natural Language Toolkit
- spaCy: Industrial-strength NLP
- scikit-learn: Machine Learning
- TensorFlow/Keras: Deep Learning
- Gensim: Topic Modeling & Word Embeddings

---


---

## 📄 License

This project is created for educational purposes as part of the NLP course (Fall 2025).

---

