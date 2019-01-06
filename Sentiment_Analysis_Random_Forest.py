# -*- coding: utf-8 -*-
"""

"""

# -*- coding: utf-8 -*-
"""

"""
# Import libraries
import pandas as pd
import re
import nltk
# Download stopwords list
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Import the dataset & initiate corpus
dataset = pd.read_csv('Sentiment_Labelled_Data_1.txt', header = None, delimiter = '\t')
corpus = []

# Cleaning the texts using for loop
for i in range(0, 2400):
# Keeping only Alphabets and spaces
    review = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
# Converting reviews to lowercase
    review = review.lower()
# Tokenisation (list of words)
    review = review.split()
# Stemming (converting each word to its root word)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
# Transforming list of words back to strings
    review = ' '.join(review)
# Appending cleaned text into corpus
    corpus.append(review)
    
# Creating the Bag of Words model ( using Tf-idf vectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
# Transforming reviews into vectors (Sparse matrix)
X = vectorizer.fit_transform(corpus).toarray()
# y is the Dependent variable
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set (80-20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Confusion Matrix and accuracy of test data analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
confMat = confusion_matrix(y_test, y_pred)
print("Confution Matrix test data = ", confMat)
print("Accuracy Test data = ", accuracy_score(y_test, y_pred))

# Sentiment analysis of Validation data
# Importing Validation data & creating Corpus for validation data
validationset = pd.read_csv('Validation.txt', header = None, delimiter = '\t')
corpus_val =[]

# Cleaning Validation Data using for loop (same cleaning process as above)
for j in range(0,600):
    review_val = re.sub('[^a-zA-Z]', ' ', validationset[0][j])
    review_val = review_val.lower()
    review_val = review_val.split()
    review_val = [ps.stem(word) for word in review_val]
    review_val = ' '.join(review_val)
    corpus_val.append(review_val)

# Vectorizing Validation data (Bag of words)
X_val = vectorizer.transform(corpus_val).toarray()
y_val = validationset.iloc[:, 1].values   

# Predicting Validation data
y_val_pred = classifier.predict(X_val)
confMat_val = confusion_matrix(y_val, y_val_pred)

print("Confution Matrix validation data = ", confMat_val)
print("Accuracy Validation data = ", (accuracy_score(y_val, y_val_pred)))
print("f1 score = ", f1_score(y_val, y_val_pred))
print("Precision = ", precision_score(y_val, y_val_pred))
print("Recall score = ", recall_score(y_val, y_val_pred))