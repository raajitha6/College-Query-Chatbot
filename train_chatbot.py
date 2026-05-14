import json
import random
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load intents
with open("intents.json") as f:
    intents=json.load(f)

# Prepare dataset
X_text=[]
y=[]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X_text.append(pattern)
        y.append(intent["tag"])

# Vectorize text
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(X_text)

# Find the best model
def find_best_model(X, y, test_size=0.2):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=100)

    models=[
        ('Logistic Regression', LogisticRegression(), {
            'penalty': ['l2'],
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear'],
            'max_iter': [100, 1000, 20000]
        }),
        ('MultinomialNB', MultinomialNB(), {
            'alpha': [0.1, 0.5, 1.0]
        }),
        ('Linear SVC', LinearSVC(), {
            'penalty': ['l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.1, 1, 10],
            'max_iter': [100, 1000, 20000]
        }),
        ('Decision Tree', DecisionTreeClassifier(), {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }),
        ('Random Forest', RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        })
    ]

    best_score=0
    best_model=None

    for name, model, param_grid in models:
        grid=GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        score=grid.score(X_test, y_test)
        print(f"{name} Accuracy: {score:.4f}")

        if score > best_score:
            best_score=score
            best_model=grid.best_estimator_

    print(f"\n✅ Best Model: {type(best_model).__name__} with accuracy {best_score:.4f}")
    return best_model

best_model=find_best_model(X, y)

# Fit best model on all data
best_model.fit(X, y)

# Save model and vectorizer
pickle.dump(best_model, open("chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("🎉 Training complete. Model and vectorizer saved.")
