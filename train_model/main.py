import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pkl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_model(data):
    feature_cols = ['Gender','Age','Income','Collateral','House_Status','Marriage','Duration','Loan History','Loan Purpose']

    X = data[feature_cols]
    y = data['Creditworthiness'].astype(int)

    # Preprocessing pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data():
    df = pd.read_excel(r'C:/Users/admin/Downloads/Data.xlsx')
    data = df.iloc[:, 1:]
    data.info()
    data.head(5)
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].apply(lambda x: str(x).split('-')[0])
            data[col] = data[col].apply(lambda x: str(x).split('.')[0])
    return data

def main():
    data = get_clean_data()
    model, scaler = create_model(data)

    with open('C:/Users/admin//Downloads/Compressed/WorkSpace/Python/LoanPrediction/train_model/model.pkl', 'wb') as f:
        pkl.dump(model, f)

    with open('C:/Users/admin//Downloads/Compressed/WorkSpace/Python/LoanPrediction/train_model/scaler.pkl', 'wb') as f:
        pkl.dump(scaler, f)

if __name__ == '__main__':
    main()
