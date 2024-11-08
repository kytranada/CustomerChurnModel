import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


class ChurnPredictor:
    def __init__(self):
        self.numerical_columns = [
            'Tenure in Months', 'Monthly Charge', 'Total Charges',
            'Age'
        ]
        self.categorical_columns = [
            'Phone Service', 'Internet Service', 'Streaming',
            'Gender'
        ]
        self.target_column = 'Churn Value'
        self.id_column = 'Customer ID'
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None  # Will store actual feature names after preprocessing

    def load_data(self, filepath='./data/processed/merged.parquet'):
        """Load the processed dataset."""
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def encode_categoricals(self, X):
        """Encode categorical variables and print category counts."""
        for col in self.categorical_columns:
            X[col] = X[col].astype('category')
            X[col] = X[col].cat.codes
            print(f"{col} has {len(X[col].unique())} unique categories")
        return X

    def analyze_features(self, X):
        """Analyze feature correlations."""
        numerical_corr = X[self.numerical_columns].corr()
        print("\nFeature Correlations:")
        print(numerical_corr)
        return numerical_corr

    def scale_features(self, X, is_training=True):
        """Scale numerical features."""
        if is_training:
            X[self.numerical_columns] = self.scaler.fit_transform(X[self.numerical_columns])
        else:
            X[self.numerical_columns] = self.scaler.transform(X[self.numerical_columns])
        return X

    def preprocess_data(self, df):
        """Complete preprocessing pipeline."""
        # Print initial class distribution to verify balance
        print("\nClass distribution:")
        print(df[self.target_column].value_counts(normalize=True))

        # Drop excluded columns
        df = df.drop(columns=['Churn Reason', 'Churn Category', 'City', 'State', 'Zip Code', 'Latitude', 'Longitude'])

        # Split features and target
        X = df.drop(columns=[self.target_column, self.id_column])
        y = df[self.target_column]

        # Store feature names in sorted order
        self.feature_names = sorted(X.columns.tolist())

        # Process features
        X = self.encode_categoricals(X)
        self.analyze_features(X)
        X = self.scale_features(X)

        # Ensure columns are in the same order as feature_names
        X = X[self.feature_names]

        # Split the data with stratification to maintain balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def get_model_params(self):
        """Define model parameters for grid search."""
        return {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'scale_pos_weight': [1]
        }

    def train(self, X_train, y_train):
        """Train the model using GridSearchCV."""
        base_model = XGBClassifier(
            random_state=42,
            enable_categorical=True,
            tree_method='hist'
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.get_model_params(),
            cv=StratifiedKFold(n_splits=4),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Print feature importance
        self.print_feature_importance()
        return self.model

    def print_feature_importance(self):
        """Print feature importance scores."""
        if self.model is not None and self.feature_names is not None:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            print("\nFeature Importances:")
            print(feature_importance.sort_values('importance', ascending=False))

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Print various metrics
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nPrediction Distribution:")
        print(pd.Series(y_pred).value_counts(normalize=True))
        print("\nProbability Distribution Statistics:")
        print(pd.DataFrame(y_pred_proba).describe())

    def save(self, model_dir='./model'):
        """Save the model and associated components."""
        if self.model is None:
            raise ValueError("No model to save!")

        # Create model directory if it doesn't exist
        import os
        os.makedirs(model_dir, exist_ok=True)

        # Save components
        joblib.dump(self.model, f'{model_dir}/churn_model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump((self.numerical_columns, self.categorical_columns),
                    f'{model_dir}/columns.pkl')
        print(f"Model and components saved to {model_dir}")

def main():
    # Initialize predictor
    predictor = ChurnPredictor()

    # Load data
    df = predictor.load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)

    # Train model
    predictor.train(X_train, y_train)

    # Evaluate model
    predictor.evaluate(X_test, y_test)

    # Save model
    predictor.save()


if __name__ == "__main__":
    main()