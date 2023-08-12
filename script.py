import os
import re
import nltk
import shutil
import pickle
import PyPDF2
import string
import argparse
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


nltk.download('punkt')
nltk.download('stopwords')

def load_encoder():
    with open('model/label_encoder_pickle.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return encoder

def load_model():
    model = keras.models.load_model("model/resume_category_model_acc_70.h5",compile=False)
    return model

def preprocess(txt):
    #convert into lower case and remove all punctuation and extra spaces.  
    txt = txt.lower()
    txt = re.sub(f"[{string.punctuation}]", " ", txt)
    txt = re.sub(f"[0-9]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    
    #tokenization and remove stop word
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(txt)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    clean_txt = ' '.join(stemmed_tokens)
 
    return clean_txt

def predict_classes(model, encoded_data, encoder):
    predictions = model.predict(encoded_data)
    predicted_classes = [encoder.classes_[i] for i in predictions.argmax(axis=1)]
    return predicted_classes

def pdf_to_string(pdf_path):
    pdf_text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    
    return pdf_text

def data_preparation(X):
        max_words = 6000
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        X = tokenizer.texts_to_sequences(X)
        max_sequence_length = 1200
        X_padded = pad_sequences(X, padding='post', maxlen=max_sequence_length)
        # print(X_padded.shape)
        X_padded = np.array(X_padded)
        return X_padded
def move_directory():
    df = pd.read_csv('categorized_resumes.csv')
    try:
        os.makedirs('CV_folder')
    except:
        pass
    for _, row in df.iterrows():
        file_path = row['filename']   
        folder_name = 'CV_folder\\'+row['category']    
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        destination_path = os.path.join(folder_name, file_name)
        shutil.copy(file_path, destination_path)
    else:
        print(f'File {file_path} does not exist')
def main(data_directory):
    id = []
    category = []
    
    # Load encoder and model
    encoder = load_encoder()
    model = load_model()
    # List files in the specified directory
    file_list = [os.path.join(data_directory, filename) for filename in os.listdir(data_directory) if filename.endswith(".pdf")]

    for file_path in file_list:
        X = []
        resume_str =  pdf_to_string(file_path)
        clean_resume = preprocess(resume_str)
        X.append(clean_resume)
        X_padded = data_preparation(X)
        predicted_classes = predict_classes(model, X_padded, encoder)

        # Print predictions
        for text, predicted_class in zip(X_padded, predicted_classes):
            print(f"Predicted Class: {predicted_class}")
            print()
        pdf_name = file_path.split('/')[-1]
        id.append(pdf_name)
        category.append(predicted_class)

    data = {
        'filename': id,
        'category': category
        }
    df = pd.DataFrame(data)
    df.to_csv('categorized_resumes.csv', index=False)
    move_directory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to the directory containing text files.") 
    args = parser.parse_args()

    main(args.dir)