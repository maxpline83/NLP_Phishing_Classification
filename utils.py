from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

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

def evaluate_model(model, X_test, y_test):

    # Make predictions on test data
    prediction_proba = model.predict(X_test)
    prediction = (prediction_proba > 0.5).astype(int)
    # Create a heatmap for the confusion matrix
    conf_matrix = confusion_matrix(y_test, prediction)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classficiation report: \n {}".format(classification_report(y_test, prediction)))
    return classification_report(y_test, prediction, output_dict=True)

def tokenize_training_data(X_train, max_words = 1000, max_sequence_length = 32):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return tokenizer, sequences

def word2vec_training_data(X_train, vector_size=100, min_count=1, window=5):
    # Tokenize the input text
    X_train_tokenized = [text_to_word_sequence(text) for text in X_train]

    # Initialize and train the Word2Vec model
    model = Word2Vec(sentences=X_train_tokenized, vector_size=vector_size, min_count=min_count, window=window)

    # Convert the tokenized text to vectors
    vectors = [model.wv[text] for text in X_train_tokenized]

    return model, vectors