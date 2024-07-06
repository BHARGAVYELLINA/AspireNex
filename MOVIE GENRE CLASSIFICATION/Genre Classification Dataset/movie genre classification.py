from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC 
from sklearn.metrics import classification_report
import numpy as np

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split(' ::: ')
        if len(parts) == 4:
            data.append((parts[0], parts[1], parts[2], parts[3]))
    return data

train_data = load_data('MOVIE GENRE CLASSIFICATION\Genre Classification Dataset\train_data.txt')
test_data = load_data('MOVIE GENRE CLASSIFICATION\Genre Classification Dataset\\test_data.txt')

train_texts = [item[3] for item in train_data]
train_labels = [item[2] for item in train_data]
test_texts = [item[2] + ' ' + item[3] for item in test_data]

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

classifier = SVC(kernel='linear')

model = Pipeline([
    ('tfidf', vectorizer),
    ('clf', classifier),
])

model.fit(train_texts, train_labels)

predicted = model.predict(test_texts)

for idx, prediction in enumerate(predicted):
    print(f"ID: {test_data[idx][0]} ::: PREDICTED_GENRE: {prediction}")