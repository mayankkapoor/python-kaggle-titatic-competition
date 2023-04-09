from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

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

@app.route('/api/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    data = pd.DataFrame(input_data, index=[0])
    data = preprocess_data(data)
    data = feature_engineering(data)
    prediction = model.predict(data)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    # make sure to copy the model.pkl file into this flask app's folder as model.pkl before running the app
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8000)
