import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# MLOps Pipeline: Iris Detection
class IrisMLPipeline:
    def __init__(self, data_path='iris.csv'):
        self.data_path = data_path
        self.model = None
        self.metrics = {}

    def load_data(self):
        # In a real MLOps env, we might use DVC or a Feature Store here
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        df = pd.read_csv(url, header=None, names=cols)
        return df

    def train(self, df):
        X = df.drop('species', axis=1)
        y = df['species']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Simple Model for demonstration
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train, y_train)
        
        # Evaluation
        predictions = self.model.predict(X_test)
        self.metrics['accuracy'] = accuracy_score(y_test, predictions)
        self.metrics['report'] = classification_report(y_test, predictions)
        
        print(f"Training Complete. Accuracy: {self.metrics['accuracy']}")
        return self.metrics

    def save_model(self, path='iris_model.pkl'):
        # Model Registry step
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    pipeline = IrisMLPipeline()
    data = pipeline.load_data()
    pipeline.train(data)
    pipeline.save_model()
