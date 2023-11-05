from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def handles_url(text):
    url_pattern = r'https?://\S+|www\.\S+'
    text_without_http = re.sub(url_pattern, '', str(text))

    html_pattern = r'<.*?>'
    text_without_html = re.sub(html_pattern, '', text_without_http)

    email_pattern = r'\S+@\S+'
    cleaned_text = re.sub(email_pattern, '', text_without_html)
    return cleaned_text

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    cleaned_text = handles_url(text)
    tokens = word_tokenize(cleaned_text)
    words = [word.lower() for word in tokens if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 1]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [re.sub(r'(.)\1{3,}', r'\1\1\1', word) for word in words]
    words = [re.sub(r'[^a-zA-Z]', '', word) for word in words]
    words = [word.replace("'s", "") for word in words]
    words = [word if word not in ["cc", "to", "cci", "subject", "regards"] else "" for word in words]
     
    return ' '.join(words)