from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np

def handles_url(text):
    url_pattern = r'https?://\S+|www\.\S+'
    cleaned_text = re.sub(url_pattern, '', str(text))
    return cleaned_text

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    cleaned_text = handles_url(text)
    tokens = word_tokenize(cleaned_text)
    words = [word.lower() for word in tokens if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def get_vectors(embeddings, words, size_of_vector=300):
    """
    Input:
        embeddings: the embeddings for the words
        words: a list of words
    Output: 
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
        
    """
    X = np.zeros((1, size_of_vector))
    for word in words:
        eng_emb = embeddings[word]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X

def Series_to_list(series):
    """
    Input:
        Series: a pandas Series
    Output: 
        a list of lists where each list is a list of words in the corresponding row of the Series        
    """
    return series.apply(lambda x: x.split()).to_list()