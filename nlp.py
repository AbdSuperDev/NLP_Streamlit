import os
import re
import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import torch
from transformers import BertModel, BertTokenizer
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer

# Téléchargements nécessaires
nltk.download('punkt')
nltk.download('stopwords')
subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])

# Initialiser SpaCy
nlp = spacy.load('fr_core_news_sm')

# Initialize French stemmer
stemmer = FrenchStemmer()

# Function for reading articles from directory
def read_articles(directory):
    articles = []
    titles = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                titles.append(lines[0].strip())
                articles.append(" ".join(lines[1:]))  # Assuming content starts from line 2
    return articles, titles

# Function for cleaning article content
def clean_content(content):
    start_pattern = re.compile(r'Sign up[\s\S]*?Share\n')
    end_pattern = re.compile(r'Written by[\s\S]*?Teams')
    content = re.sub(start_pattern, '', content)
    content = re.sub(end_pattern, '', content)
    return content

# Preprocess text function
def preprocess_text(text):
    # Tokenization with spacy
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]

    # Stemming with nltk
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens

# Read articles from local directory
directory = '.'  # Replace with your directory path containing articles
corpus, titles = read_articles(directory)

# Clean and preprocess each article
preprocessed_corpus = [preprocess_text(article) for article in corpus]

# Convert preprocessed corpus back to strings
preprocessed_corpus_str = [" ".join(article) for article in preprocessed_corpus]

# Generate keywords using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_corpus_str)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
tfidf_scores = tfidf_vectors.toarray()

top_n = 10
keywords = []
for doc_scores in tfidf_scores:
    top_indices = doc_scores.argsort()[-top_n:]
    doc_keywords = feature_names[top_indices]
    keywords.extend(doc_keywords)

keywords = list(set(keywords))

# Streamlit interface
st.title("Analyse de Similarité des Articles")

# Choose vectorization method
vectorization_method = st.selectbox("Choisir la méthode de vectorisation :", ("TF-IDF", "Bag of Words", "Word2Vec", "BERT"))

# Choose number of top documents to display
top_x = st.number_input("Nombre de documents pertinents à afficher (Top X) :", min_value=1, max_value=len(corpus), value=5, step=1)

# Vectorization and similarity calculation
if vectorization_method == "TF-IDF":
    vectorizer = TfidfVectorizer()
    article_vectors = vectorizer.fit_transform(preprocessed_corpus_str)
    keyword_vector = vectorizer.transform([" ".join(keywords)])
elif vectorization_method == "Bag of Words":
    vectorizer = CountVectorizer()
    article_vectors = vectorizer.fit_transform(preprocessed_corpus_str)
    keyword_vector = vectorizer.transform([" ".join(keywords)])
elif vectorization_method == "Word2Vec":
    word2vec_model = Word2Vec(sentences=preprocessed_corpus, vector_size=100, window=5, min_count=1, workers=4)
    article_vectors = np.array([np.mean([word2vec_model.wv[word] for word in article if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for article in preprocessed_corpus])
    keyword_vector = np.mean([word2vec_model.wv[word] for word in keywords if word in word2vec_model.wv] or [np.zeros(100)], axis=0).reshape(1, -1)
elif vectorization_method == "BERT":
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    article_vectors = torch.stack([get_bert_embedding(" ".join(article)) for article in preprocessed_corpus])
    keyword_vector = get_bert_embedding(" ".join(keywords)).unsqueeze(0)


# Calculate cosine similarity
similarities = cosine_similarity(article_vectors, keyword_vector)

# Display top similar articles
st.write(f"Top {top_x} articles par similarité {vectorization_method} :")
top_n_indices = similarities.squeeze().argsort()[::-1][:top_x]
for idx in top_n_indices:
    st.write(f"Article {idx + 1}: {titles[idx]}")
