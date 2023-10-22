import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def Logistic_Regression(tfidf_train, y_train, tfidf_test, y_test):
    model = LogisticRegression()
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)
    return accuracy_score(y_test, y_pred)

def PAC(tfidf_train, y_train, tfidf_test, y_test):
    model = PassiveAggressiveClassifier(max_iter=100)
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)
    return accuracy_score(y_test, y_pred)

def Naive_Bayes(tfidf_train, y_train, tfidf_test, y_test):
    model = MultinomialNB()
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)
    return accuracy_score(y_test, y_pred)

def Random_Forest(tfidf_train, y_train, tfidf_test, y_test):
    model = RandomForestClassifier()
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)
    return accuracy_score(y_test, y_pred)

def SVM(tfidf_train, y_train, tfidf_test, y_test):
    model = SVC()
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)
    return accuracy_score(y_test, y_pred)

if __name__=='__main__':
    df = pd.read_csv('data.csv')
    labels = df.label

    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)

    score = Logistic_Regression(tfidf_train, y_train, tfidf_test, y_test)
    print(f'Accuracy for Logistic Regression: {round(score*100,2)}%')

    score = PAC(tfidf_train, y_train, tfidf_test, y_test)
    print(f'Accuracy for PassiveAggresive Classifier: {round(score*100,2)}%')

    score = Naive_Bayes(tfidf_train, y_train, tfidf_test, y_test)
    print(f'Accuracy for Naive Bayes: {round(score*100,2)}%')

    score = Random_Forest(tfidf_train, y_train, tfidf_test, y_test)
    print(f'Accuracy for Random Forest: {round(score*100,2)}%')

    score = SVM(tfidf_train, y_train, tfidf_test, y_test)
    print(f'Accuracy for SVM: {round(score*100,2)}%')

