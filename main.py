"""
Mental Health Social Media Classifier
NLP Course Project - Fall 2025

A comprehensive NLP system for detecting mental health concerns in social media posts
using advanced preprocessing, feature extraction, deep learning, and interpretability techniques.
"""

# ============================================================================
# PART 1: IMPORTS AND SETUP
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
import spacy
nltk.download('all')
# Sentiment Analysis
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_recall_fscore_support,
                            roc_curve, auc, roc_auc_score)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Embedding, LSTM, Bidirectional,
                                     Dropout, Conv1D, MaxPooling1D,
                                     GlobalMaxPooling1D, Attention, Input,
                                     Concatenate, BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Word Embeddings
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

print("All libraries imported successfully!")

# ============================================================================
# PART 2: DATA LOADING AND EXPLORATION
# ============================================================================

def load_and_explore_data():
    """Load datasets and perform initial exploration"""

    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("="*80)

    # Load datasets (update paths as needed)
    # Dataset 1: Mental Health Dataset
    try:
        df1 = pd.read_csv('Mental_Health_Dataset.csv')
        print(f"\nDataset 1 loaded: {df1.shape}")
        print(f"Columns: {df1.columns.tolist()}")
    except:
        print("\n⚠️ Mental Health Dataset not found. Using sample data.")
        df1 = pd.DataFrame({
            'text': ['I feel so depressed today', 'Anxiety is killing me',
                    'Having a great day!', 'Life is beautiful'],
            'label': [1, 1, 0, 0]
        })

    # Dataset 2: Twitter Depression Dataset
    try:
        df2 = pd.read_csv('twitter_depression.csv')
        print(f"Dataset 2 loaded: {df2.shape}")
        print(f"Columns: {df2.columns.tolist()}")
    except:
        print("\n⚠️ Twitter Depression Dataset not found. Using sample data.")
        df2 = pd.DataFrame({
            'text': ['I cant take this anymore', 'Feeling hopeless',
                    'Everything is awesome', 'Love my life'],
            'label': [1, 1, 0, 0]
        })

    # Combine datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Ensure consistent column names
    if 'text' not in df.columns:
        text_col = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower() or 'post' in col.lower()][0]
        df.rename(columns={text_col: 'text'}, inplace=True)

    if 'label' not in df.columns:
        label_col = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower()][0]
        df.rename(columns={label_col: 'label'}, inplace=True)

    # Basic cleaning
    df = df[['text', 'label']].dropna()
    df = df.drop_duplicates()

    # Convert labels to binary (0: Normal, 1: Mental Health Concern)
    df['label'] = df['label'].apply(lambda x: 1 if x in [1, '1', 'depression', 'anxiety'] else 0)

    print(f"\n✓ Combined dataset shape: {df.shape}")
    print(f"\n✓ Class distribution:\n{df['label'].value_counts()}")
    print(f"\n✓ Sample texts:")
    print(df.head())

    return df

# ============================================================================
# PART 3: ADVANCED TEXT PREPROCESSING
# ============================================================================

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

def apply_preprocessing(df):
    """Apply preprocessing to entire dataset"""

    print("\n" + "="*80)
    print("STEP 2: TEXT PREPROCESSING")
    print("="*80)

    preprocessor = AdvancedTextPreprocessor()

    print("\nApplying preprocessing pipeline...")
    df['tokens'] = df['text'].apply(lambda x: preprocessor.preprocess(x)[0])
    df['cleaned_text'] = df['text'].apply(lambda x: preprocessor.preprocess(x)[1])

    print(f"✓ Preprocessing complete!")
    print(f"\nExample transformation:")
    print(f"Original: {df['text'].iloc[0][:100]}")
    print(f"Cleaned: {df['cleaned_text'].iloc[0][:100]}")
    print(f"Tokens: {df['tokens'].iloc[0][:10]}")

    return df, preprocessor

# ============================================================================
# PART 4: LINGUISTIC ANALYSIS
# ============================================================================

def perform_linguistic_analysis(df):
    """POS tagging, dependency parsing, and chunking"""

    print("\n" + "="*80)
    print("STEP 3: LINGUISTIC ANALYSIS")
    print("="*80)

    # POS Tagging
    print("\n→ Performing POS tagging...")
    df['pos_tags'] = df['tokens'].apply(lambda tokens: pos_tag(tokens))

    # Extract POS patterns
    def extract_pos_features(pos_tags):
        pos_counts = Counter([tag for word, tag in pos_tags])
        return {
            'num_nouns': pos_counts['NN'] + pos_counts['NNS'] + pos_counts['NNP'],
            'num_verbs': pos_counts['VB'] + pos_counts['VBD'] + pos_counts['VBG'],
            'num_adjectives': pos_counts['JJ'] + pos_counts['JJR'] + pos_counts['JJS'],
            'num_adverbs': pos_counts['RB'] + pos_counts['RBR'] + pos_counts['RBS']
        }

    pos_features = df['pos_tags'].apply(extract_pos_features)
    df['num_nouns'] = pos_features.apply(lambda x: x['num_nouns'])
    df['num_verbs'] = pos_features.apply(lambda x: x['num_verbs'])
    df['num_adjectives'] = pos_features.apply(lambda x: x['num_adjectives'])

    print("✓ POS tagging complete")

    # Dependency Parsing using spaCy (sample)
    print("\n→ Performing dependency parsing (sample)...")
    sample_texts = df['text'].head(3).tolist()

    for i, text in enumerate(sample_texts):
        doc = nlp(text[:200])  # Limit length
        print(f"\nText {i+1}: {text[:100]}...")
        print("Dependencies:")
        for token in doc[:10]:
            print(f"  {token.text:15} -> {token.dep_:10} -> {token.head.text}")

    # Chunking (Noun Phrases)
    print("\n→ Performing chunking...")
    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}
        VP: {<VB.*><NP|PP>}
        PP: {<IN><NP>}
    """
    chunk_parser = RegexpParser(grammar)

    def extract_chunks(pos_tags):
        tree = chunk_parser.parse(pos_tags)
        chunks = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                chunks.append(' '.join([word for word, tag in subtree.leaves()]))
        return chunks

    df['noun_phrases'] = df['pos_tags'].apply(extract_chunks)

    print("✓ Linguistic analysis complete")

    return df

# ============================================================================
# PART 5: FREQUENCY ANALYSIS
# ============================================================================

def frequency_analysis(df):
    """Analyze word frequencies and create visualizations"""

    print("\n" + "="*80)
    print("STEP 4: FREQUENCY ANALYSIS")
    print("="*80)

    # Overall word frequency
    all_tokens = [token for tokens in df['tokens'] for token in tokens]
    word_freq = Counter(all_tokens)

    print(f"\n✓ Total unique words: {len(word_freq)}")
    print(f"✓ Top 20 most common words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word:15} : {count:5}")

    # Class-specific analysis
    mental_health_tokens = [token for tokens, label in zip(df['tokens'], df['label'])
                           if label == 1 for token in tokens]
    normal_tokens = [token for tokens, label in zip(df['tokens'], df['label'])
                    if label == 0 for token in tokens]

    mh_freq = Counter(mental_health_tokens)
    normal_freq = Counter(normal_tokens)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Overall frequency
    top_words = [word for word, _ in word_freq.most_common(20)]
    top_counts = [count for _, count in word_freq.most_common(20)]
    axes[0, 0].barh(top_words, top_counts, color='steelblue')
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].set_title('Top 20 Most Common Words (Overall)')
    axes[0, 0].invert_yaxis()

    # Mental health posts
    mh_words = [word for word, _ in mh_freq.most_common(15)]
    mh_counts = [count for _, count in mh_freq.most_common(15)]
    axes[0, 1].barh(mh_words, mh_counts, color='crimson')
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].set_title('Top Words in Mental Health Posts')
    axes[0, 1].invert_yaxis()

    # Normal posts
    normal_words = [word for word, _ in normal_freq.most_common(15)]
    normal_counts = [count for _, count in normal_freq.most_common(15)]
    axes[1, 0].barh(normal_words, normal_counts, color='seagreen')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title('Top Words in Normal Posts')
    axes[1, 0].invert_yaxis()

    # Word length distribution
    word_lengths = [len(token) for tokens in df['tokens'] for token in tokens]
    axes[1, 1].hist(word_lengths, bins=20, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Word Length')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Word Length Distribution')

    plt.tight_layout()
    plt.savefig('frequency_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Frequency analysis visualization saved as 'frequency_analysis.png'")

    return word_freq, mh_freq, normal_freq

# ============================================================================
# PART 6: SENTIMENT ANALYSIS
# ============================================================================

def sentiment_analysis(df):
    """Perform sentiment analysis using VADER and TextBlob"""

    print("\n" + "="*80)
    print("STEP 5: SENTIMENT ANALYSIS")
    print("="*80)

    # VADER Sentiment
    print("\n→ Applying VADER sentiment analysis...")
    sia = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        scores = sia.polarity_scores(text)
        return scores

    vader_scores = df['text'].apply(get_vader_sentiment)
    df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])

    # TextBlob Sentiment
    print("→ Applying TextBlob sentiment analysis...")

    def get_textblob_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    textblob_scores = df['text'].apply(get_textblob_sentiment)
    df['textblob_polarity'] = textblob_scores.apply(lambda x: x[0])
    df['textblob_subjectivity'] = textblob_scores.apply(lambda x: x[1])

    print("✓ Sentiment analysis complete")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # VADER Compound by Class
    df_mh = df[df['label'] == 1]
    df_normal = df[df['label'] == 0]

    axes[0, 0].hist([df_mh['vader_compound'], df_normal['vader_compound']],
                    bins=30, label=['Mental Health', 'Normal'],
                    color=['crimson', 'seagreen'], alpha=0.7)
    axes[0, 0].set_xlabel('VADER Compound Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Sentiment Distribution by Class (VADER)')
    axes[0, 0].legend()

    # TextBlob Polarity by Class
    axes[0, 1].hist([df_mh['textblob_polarity'], df_normal['textblob_polarity']],
                    bins=30, label=['Mental Health', 'Normal'],
                    color=['crimson', 'seagreen'], alpha=0.7)
    axes[0, 1].set_xlabel('TextBlob Polarity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sentiment Distribution by Class (TextBlob)')
    axes[0, 1].legend()

    # Scatter: Polarity vs Subjectivity
    axes[1, 0].scatter(df_mh['textblob_polarity'], df_mh['textblob_subjectivity'],
                      alpha=0.5, c='crimson', label='Mental Health', s=20)
    axes[1, 0].scatter(df_normal['textblob_polarity'], df_normal['textblob_subjectivity'],
                      alpha=0.5, c='seagreen', label='Normal', s=20)
    axes[1, 0].set_xlabel('Polarity')
    axes[1, 0].set_ylabel('Subjectivity')
    axes[1, 0].set_title('Polarity vs Subjectivity')
    axes[1, 0].legend()

    # Box plots
    sentiment_data = pd.DataFrame({
        'Class': ['Mental Health']*len(df_mh) + ['Normal']*len(df_normal),
        'Compound': list(df_mh['vader_compound']) + list(df_normal['vader_compound'])
    })
    sns.boxplot(data=sentiment_data, x='Class', y='Compound', ax=axes[1, 1],
                palette={'Mental Health': 'crimson', 'Normal': 'seagreen'})
    axes[1, 1].set_title('VADER Compound Score by Class')

    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Sentiment visualization saved as 'sentiment_analysis.png'")

    return df

# ============================================================================
# PART 7: FEATURE EXTRACTION (TF-IDF, Word2Vec)
# ============================================================================

def feature_extraction(df):
    """Extract features using TF-IDF and Word2Vec"""

    print("\n" + "="*80)
    print("STEP 6: FEATURE EXTRACTION")
    print("="*80)

    # TF-IDF Features
    print("\n→ Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_text'])

    print(f"✓ TF-IDF matrix shape: {tfidf_features.shape}")

    # Get top TF-IDF terms
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_features.toarray().sum(axis=0)
    top_tfidf_idx = tfidf_scores.argsort()[-20:][::-1]

    print(f"\n✓ Top 20 TF-IDF terms:")
    for idx in top_tfidf_idx:
        print(f"  {feature_names[idx]:20} : {tfidf_scores[idx]:.2f}")

    # Word2Vec Features
    print("\n→ Training Word2Vec model...")
    sentences = df['tokens'].tolist()

    # Detect phrases (bigrams)
    phrases = Phrases(sentences, min_count=5, threshold=10)
    phraser = Phraser(phrases)
    sentences_with_phrases = [phraser[sent] for sent in sentences]

    # Train Word2Vec
    w2v_model = Word2Vec(sentences=sentences_with_phrases,
                        vector_size=100,
                        window=5,
                        min_count=2,
                        workers=4,
                        epochs=10)

    print(f"✓ Word2Vec trained with vocabulary size: {len(w2v_model.wv)}")

    # Create document embeddings (average word vectors)
    def get_avg_w2v(tokens, model):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    df['w2v_embedding'] = df['tokens'].apply(lambda x: get_avg_w2v(x, w2v_model))
    w2v_features = np.vstack(df['w2v_embedding'].values)

    print(f"✓ Word2Vec embedding matrix shape: {w2v_features.shape}")

    # Test word similarities
    print("\n✓ Word2Vec similarity examples:")
    test_words = ['sad', 'happy', 'anxiety', 'depression']
    for word in test_words:
        if word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(word, topn=5)
            print(f"\n  Most similar to '{word}':")
            for sim_word, score in similar:
                print(f"    {sim_word:15} : {score:.3f}")

    return tfidf_vectorizer, tfidf_features, w2v_model, w2v_features

# ============================================================================
# PART 8: TOPIC CLUSTERING (LDA & K-Means)
# ============================================================================

def topic_clustering(df, tfidf_features):
    """Perform topic modeling using LDA and K-Means clustering"""

    print("\n" + "="*80)
    print("STEP 7: TOPIC CLUSTERING")
    print("="*80)

    # LDA Topic Modeling
    print("\n→ Performing LDA topic modeling...")
    count_vectorizer = CountVectorizer(max_features=1000, max_df=0.8, min_df=2)
    count_features = count_vectorizer.fit_transform(df['cleaned_text'])

    n_topics = 5
    lda_model = LatentDirichletAllocation(n_components=n_topics,
                                         random_state=42,
                                         max_iter=20)
    lda_topics = lda_model.fit_transform(count_features)

    # Display topics
    feature_names = count_vectorizer.get_feature_names_out()
    print(f"\n✓ Discovered {n_topics} topics:")

    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

    df['dominant_topic'] = lda_topics.argmax(axis=1)

    # K-Means Clustering on TF-IDF features
    print("\n→ Performing K-Means clustering...")

    # Reduce dimensionality for clustering
   # Automatically cap n_components to the number of tfidf features
    n_features = tfidf_features.shape[1]
    n_components = min(50, n_features)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_features = svd.fit_transform(tfidf_features)


    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(reduced_features)

    print(f"✓ K-Means clustering complete")
    print(f"\n✓ Cluster distribution:")
    print(df['cluster'].value_counts().sort_index())

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Topic distribution
    topic_dist = df['dominant_topic'].value_counts().sort_index()
    axes[0].bar(range(len(topic_dist)), topic_dist.values, color='steelblue')
    axes[0].set_xlabel('Topic')
    axes[0].set_ylabel('Number of Documents')
    axes[0].set_title('LDA Topic Distribution')
    axes[0].set_xticks(range(len(topic_dist)))
    axes[0].set_xticklabels([f'Topic {i+1}' for i in range(len(topic_dist))])

    # Cluster visualization (2D projection)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    reduced_2d = pca.fit_transform(reduced_features)

    scatter = axes[1].scatter(reduced_2d[:, 0], reduced_2d[:, 1],
                             c=df['cluster'], cmap='viridis',
                             alpha=0.6, s=30)
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    axes[1].set_title('K-Means Clustering (PCA Projection)')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig('topic_clustering.png', dpi=300, bbox_inches='tight')
    print("\n✓ Clustering visualization saved as 'topic_clustering.png'")

    return lda_model, kmeans, df

# ============================================================================
# PART 9: CLASSICAL MACHINE LEARNING MODELS
# ============================================================================

def train_classical_models(X_train, X_test, y_train, y_test):
    """Train and evaluate classical ML models"""

    print("\n" + "="*80)
    print("STEP 8: CLASSICAL MACHINE LEARNING MODELS")
    print("="*80)

    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n→ Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

    return results

# ============================================================================
# PART 10: DEEP LEARNING MODEL (LSTM with Attention)
# ============================================================================

def build_deep_learning_model(df, max_words=10000, max_len=100):
    """Build and train a deep learning model from scratch"""

    print("\n" + "="*80)
    print("STEP 9: DEEP LEARNING MODEL (LSTM WITH ATTENTION)")
    print("="*80)

    # Tokenization for deep learning
    print("\n→ Preparing sequences...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_text'])

    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = df['label'].values

    print(f"✓ Sequence matrix shape: {X.shape}")
    print(f"✓ Vocabulary size: {len(tokenizer.word_index)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Build advanced LSTM model with attention
    print("\n→ Building Bidirectional LSTM with Attention...")

    # Input layer
    inputs = Input(shape=(max_len,))

    # Embedding layer
    embedding = Embedding(input_dim=max_words, output_dim=128,
                         input_length=max_len)(inputs)

    # Bidirectional LSTM layers
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(embedding)
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(lstm1)

    # Attention mechanism
    attention = tf.keras.layers.Attention()([lstm2, lstm2])

    # Global pooling
    pooled = GlobalMaxPooling1D()(attention)

    # Dense layers
    dense1 = Dense(64, activation='relu')(pooled)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(dropout2)

    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.AUC(name='auc')])

    print(model.summary())

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train model
    print("\n→ Training deep learning model...")
    history = model.fit(X_train, y_train,
                       validation_split=0.2,
                       epochs=30,
                       batch_size=32,
                       callbacks=[early_stop, reduce_lr],
                       verbose=1)

    # Evaluate
    print("\n→ Evaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n✓ Deep Learning Model Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Save model
    model.save('mental_health_lstm_model.h5')
    print("\n✓ Model saved as 'mental_health_lstm_model.h5'")

    return model, history, tokenizer, y_test, y_pred, y_pred_proba

# ============================================================================
# PART 11: MODEL EVALUATION AND VISUALIZATION
# ============================================================================

def comprehensive_evaluation(y_test, predictions_dict, dl_results):
    """Comprehensive model evaluation with visualizations"""

    print("\n" + "="*80)
    print("STEP 10: COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Combine all results
    all_results = {}

    # Classical models
    for name, results in predictions_dict.items():
        all_results[name] = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'y_pred': results['predictions']
        }

    # Deep learning model
    y_test_dl, y_pred_dl, y_pred_proba_dl = dl_results
    accuracy = accuracy_score(y_test_dl, y_pred_dl)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_dl, y_pred_dl, average='binary')

    all_results['LSTM + Attention'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred_dl,
        'y_pred_proba': y_pred_proba_dl
    }

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(all_results.keys()),
        'Accuracy': [v['accuracy'] for v in all_results.values()],
        'Precision': [v['precision'] for v in all_results.values()],
        'Recall': [v['recall'] for v in all_results.values()],
        'F1-Score': [v['f1'] for v in all_results.values()]
    })

    print("\n✓ Model Comparison:")
    print(comparison_df.to_string(index=False))

    # Visualizations
    fig = plt.figure(figsize=(20, 12))

    # 1. Model Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(comparison_df))
    width = 0.2

    ax1.bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='steelblue')
    ax1.bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='crimson')
    ax1.bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='seagreen')
    ax1.bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score', color='orange')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Confusion Matrix - Best Model (LSTM)
    ax2 = plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test_dl, y_pred_dl)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Normal', 'Mental Health'],
                yticklabels=['Normal', 'Mental Health'])
    ax2.set_title('Confusion Matrix (LSTM + Attention)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # 3. ROC Curve
    ax3 = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_test_dl, y_pred_proba_dl)
    roc_auc = auc(fpr, tpr)

    ax3.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve (LSTM + Attention)', fontsize=12, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(alpha=0.3)

    # 4. F1-Score Comparison
    ax4 = plt.subplot(2, 3, 4)
    colors = ['steelblue', 'crimson', 'seagreen', 'orange']
    ax4.barh(comparison_df['Model'], comparison_df['F1-Score'], color=colors)
    ax4.set_xlabel('F1-Score')
    ax4.set_title('F1-Score by Model', fontsize=12, fontweight='bold')
    ax4.set_xlim([0, 1])
    for i, v in enumerate(comparison_df['F1-Score']):
        ax4.text(v + 0.01, i, f'{v:.3f}', va='center')
    ax4.grid(axis='x', alpha=0.3)

    # 5. Precision-Recall Trade-off
    ax5 = plt.subplot(2, 3, 5)
    models_list = comparison_df['Model'].tolist()
    precisions = comparison_df['Precision'].tolist()
    recalls = comparison_df['Recall'].tolist()

    for i, model in enumerate(models_list):
        ax5.scatter(recalls[i], precisions[i], s=200, alpha=0.6,
                   c=[colors[i]], label=model)
        ax5.annotate(model, (recalls[i], precisions[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])

    # 6. Classification Report Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    report = classification_report(y_test_dl, y_pred_dl,
                                   target_names=['Normal', 'Mental Health'],
                                   output_dict=True)

    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(3)

    table = ax6.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     rowLabels=report_df.index,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15]*len(report_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(report_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Detailed Classification Report (LSTM)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive evaluation saved as 'comprehensive_evaluation.png'")

    return comparison_df

# ============================================================================
# PART 12: TRAINING HISTORY VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training history for deep learning model"""

    print("\n→ Creating training history visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history saved as 'training_history.png'")

# ============================================================================
# PART 13: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def feature_importance_analysis(tfidf_vectorizer, rf_model, X_train_tfidf):
    """Analyze feature importance from Random Forest"""

    print("\n" + "="*80)
    print("STEP 11: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n✓ Top 30 Most Important Features:")
    print(importance_df.head(30).to_string(index=False))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 20 features
    top_20 = importance_df.head(20)
    axes[0].barh(top_20['feature'], top_20['importance'], color='steelblue')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_title('Top 20 Most Important Features', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()

    # Cumulative importance
    importance_df['cumulative'] = importance_df['importance'].cumsum()
    axes[1].plot(range(len(importance_df)), importance_df['cumulative'], linewidth=2)
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Cumulative Importance')
    axes[1].set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance visualization saved as 'feature_importance.png'")

    return importance_df

# ============================================================================
# PART 14: ERROR ANALYSIS
# ============================================================================

def error_analysis(df_test, y_test, y_pred, texts_test):
    """Analyze misclassified examples"""

    print("\n" + "="*80)
    print("STEP 12: ERROR ANALYSIS")
    print("="*80)

    # Find misclassified examples
    errors_idx = np.where(y_test != y_pred)[0]

    false_positives = [i for i in errors_idx if y_pred[i] == 1 and y_test[i] == 0]
    false_negatives = [i for i in errors_idx if y_pred[i] == 0 and y_test[i] == 1]

    print(f"\n✓ Total misclassifications: {len(errors_idx)}")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")

    # Show examples
    print("\n✓ False Positive Examples (predicted mental health, actually normal):")
    for i, idx in enumerate(false_positives[:3]):
        print(f"\n  Example {i+1}:")
        print(f"  Text: {texts_test.iloc[idx][:200]}...")

    print("\n✓ False Negative Examples (predicted normal, actually mental health):")
    for i, idx in enumerate(false_negatives[:3]):
        print(f"\n  Example {i+1}:")
        print(f"  Text: {texts_test.iloc[idx][:200]}...")

    # Error distribution visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    error_types = ['False Positives', 'False Negatives', 'Correct']
    error_counts = [len(false_positives), len(false_negatives),
                   len(y_test) - len(errors_idx)]
    colors_err = ['crimson', 'orange', 'seagreen']

    axes[0].bar(error_types, error_counts, color=colors_err)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')

    # Pie chart
    axes[1].pie(error_counts, labels=error_types, autopct='%1.1f%%',
               colors=colors_err, startangle=90)
    axes[1].set_title('Prediction Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Error analysis saved as 'error_analysis.png'")

# ============================================================================
# PART 15: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""

    print("\n" + "="*80)
    print("MENTAL HEALTH SOCIAL MEDIA CLASSIFIER")
    print("NLP Course Project - Fall 2025")
    print("="*80)

    # Step 1: Load data
    df = load_and_explore_data()

    # Step 2: Preprocessing
    df, preprocessor = apply_preprocessing(df)

    # Step 3: Linguistic analysis
    df = perform_linguistic_analysis(df)

    # Step 4: Frequency analysis
    word_freq, mh_freq, normal_freq = frequency_analysis(df)

    # Step 5: Sentiment analysis
    df = sentiment_analysis(df)

    # Step 6: Feature extraction
    tfidf_vectorizer, tfidf_features, w2v_model, w2v_features = feature_extraction(df)

    # Step 7: Topic clustering
    lda_model, kmeans, df = topic_clustering(df, tfidf_features)

    # Step 8: Prepare data for classification
    print("\n" + "="*80)
    print("PREPARING DATA FOR CLASSIFICATION")
    print("="*80)

    # Combine sentiment features with text
    sentiment_features = df[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
                            'textblob_polarity', 'textblob_subjectivity',
                            'num_nouns', 'num_verbs', 'num_adjectives']].values

    # Split data for TF-IDF based models
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        tfidf_features, df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    print(f"✓ Training set: {X_train_tfidf.shape}")
    print(f"✓ Test set: {X_test_tfidf.shape}")

    # Step 9: Train classical models
    classical_results = train_classical_models(X_train_tfidf, X_test_tfidf,
                                              y_train, y_test)

    # Step 10: Train deep learning model
    dl_model, history, tokenizer, y_test_dl, y_pred_dl, y_pred_proba_dl = \
        build_deep_learning_model(df)

    # Step 11: Plot training history
    plot_training_history(history)

    # Step 12: Comprehensive evaluation
    comparison_df = comprehensive_evaluation(
        y_test, classical_results, (y_test_dl, y_pred_dl, y_pred_proba_dl)
    )

    # Step 13: Feature importance
    importance_df = feature_importance_analysis(
        tfidf_vectorizer,
        classical_results['Random Forest']['model'],
        X_train_tfidf
    )

    # Step 14: Error analysis
    test_indices = df.index[-len(y_test_dl):]
    df_test = df.loc[test_indices]
    error_analysis(df_test, y_test_dl, y_pred_dl, df_test['text'])

    # Step 15: Save artifacts
    print("\n" + "="*80)
    print("SAVING MODELS AND ARTIFACTS")
    print("="*80)

    # Save preprocessor and vectorizer
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Save best classical model
    best_classical = max(classical_results.items(),
                        key=lambda x: x[1]['f1'])[1]['model']
    with open('best_classical_model.pkl', 'wb') as f:
        pickle.dump(best_classical, f)

    # Save Word2Vec model
    w2v_model.save('word2vec_model.bin')

    print("✓ All models and artifacts saved!")

    # Final summary
    print("\n" + "="*80)
    print("PROJECT SUMMARY")
    print("="*80)

    print("\n✓ Techniques Applied:")
    print("  • Tokenization & Lemmatization (NLTK, spaCy)")
    print("  • Stopwords Removal")
    print("  • POS Tagging, Dependency Parsing, Chunking")
    print("  • Frequency Analysis")
    print("  • Sentiment Analysis (VADER, TextBlob)")
    print("  • Feature Extraction (TF-IDF, Word2Vec)")
    print("  • Topic Modeling (LDA, K-Means)")
    print("  • Classification (Naive Bayes, Logistic Regression, Random Forest)")
    print("  • Deep Learning (Bidirectional LSTM with Attention)")
    print("  • Comprehensive Evaluation & Visualization")

    print("\n✓ Best Model Performance:")
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    print(f"  Model: {best_model['Model']}")
    print(f"  Accuracy: {best_model['Accuracy']:.4f}")
    print(f"  Precision: {best_model['Precision']:.4f}")
    print(f"  Recall: {best_model['Recall']:.4f}")
    print(f"  F1-Score: {best_model['F1-Score']:.4f}")

    print("\n✓ Generated Visualizations:")
    print("  • frequency_analysis.png")
    print("  • sentiment_analysis.png")
    print("  • topic_clustering.png")
    print("  • training_history.png")
    print("  • comprehensive_evaluation.png")
    print("  • feature_importance.png")
    print("  • error_analysis.png")

    print("\n" + "="*80)
    print("PROJECT COMPLETE!")
    print("="*80)

    return df, comparison_df, dl_model, tokenizer

# ============================================================================
# PART 16: PREDICTION FUNCTION FOR NEW TEXT
# ============================================================================

def predict_mental_health(text, model, tokenizer, preprocessor, max_len=100):
    """Predict mental health concern for new text"""

    # Preprocess
    tokens, cleaned = preprocessor.preprocess(text)

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Predict
    prediction_proba = model.predict(padded, verbose=0)[0][0]
    prediction = 1 if prediction_proba > 0.5 else 0

    return {
        'text': text,
        'prediction': 'Mental Health Concern' if prediction == 1 else 'Normal',
        'confidence': prediction_proba if prediction == 1 else 1 - prediction_proba,
        'cleaned_text': cleaned
    }

# ============================================================================
# RUN THE PROJECT
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    df, comparison_df, dl_model, tokenizer = main()

    # Example predictions
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    test_texts = [
        "I'm feeling really depressed and anxious today, can't stop crying",
        "Had an amazing day at the beach with friends! Life is good!",
        "I feel so hopeless and alone, nobody understands me",
        "Just got promoted at work! Couldn't be happier!"
    ]

    preprocessor = AdvancedTextPreprocessor()

    for text in test_texts:
        result = predict_mental_health(text, dl_model, tokenizer, preprocessor)
        print(f"\nText: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("-" * 80)