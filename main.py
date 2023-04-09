import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)
import matplotlib.pyplot as plt

def preprocess_data(data):
    data = data.drop(['Ticket'], axis=1)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    return data
    
def feature_engineering(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    data['Deck'] = data['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
    data['Deck'] = data['Deck'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], 'Known')
    
    le = LabelEncoder()
    data['Title'] = le.fit_transform(data['Title'])
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])
    data['Deck'] = le.fit_transform(data['Deck'])
    data = data.drop(['Name', 'PassengerId', 'Cabin'], axis=1)
    return data

def plot_cm(model, X_test, y_test):
    plt.figure()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main():
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    train_data = feature_engineering(train_data)

    passenger_ids = test_data['PassengerId']
    test_data = feature_engineering(test_data)

    X = train_data.drop('Survived', axis=1)
    y = train_data['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Visualize the fit using a confusion matrix
    plot_cm(model, X_val, y_val)

    # Calculate and print performance metrics
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    predictions = model.predict(test_data)

    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
