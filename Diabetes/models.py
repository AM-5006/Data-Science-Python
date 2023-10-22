import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import VotingClassifier

def Decesion_Tree(X_train, X_test, y_train, y_test):
    best_features = SelectKBest(score_func=chi2, k=1)
    X_train_new = best_features.fit_transform(X_train, y_train)
    X_test_new = best_features.transform(X_test)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_new, y_train)
    y_pred = clf.predict(X_test_new)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def Logistic(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(solver='newton-cg', max_iter=1500, random_state=42)  
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def SVM(X_train, X_test, y_train, y_test):
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def KNN(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def Naives_Bayes(X_train, X_test, y_train, y_test):
    best_features = SelectKBest(score_func=chi2, k=1)
    X_train_new = best_features.fit_transform(X_train, y_train)
    X_test_new = best_features.transform(X_test)
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train_new, y_train)
    y_pred = nb_classifier.predict(X_test_new)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


if __name__=='__main__':
    df = pd.read_csv('data.csv')
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    voting_classifier = VotingClassifier(estimators=[
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Naive Bayes', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('Logistic Regresion', LogisticRegression(solver='newton-cg', max_iter=1500, random_state=42) )
    ], voting='soft') 
    
    voting_classifier.fit(X_train, y_train)
    ensemble_predictions = voting_classifier.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    print(f"Ensemble Accuracy: {ensemble_accuracy}")
    # print(f"Decision Tree :- {Decesion_Tree(X_train, X_test, y_train, y_test)}")
    # print(f"Logistic Regression :- {Logistic(X_train, X_test, y_train, y_test)}")
    # print(f"SVM :- {SVM(X_train, X_test, y_train, y_test)}")
    # print(f"KNN :- {KNN(X_train, X_test, y_train, y_test)}")
    # print(f"Naives Bayes :- {Naives_Bayes(X_train, X_test, y_train, y_test)}")
