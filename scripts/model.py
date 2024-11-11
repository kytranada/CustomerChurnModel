import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Define columns
numerical_columns = [
    'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Age'
]
categorical_columns = [
    'Phone Service', 'Internet Service', 'Streaming', 'Gender'
]
target_column = 'Churn Value'
id_column = 'Customer ID'
scaler = StandardScaler()

def load_data(filepath='./data/processed/merged.parquet'):
    """Load data from a parquet file."""
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def encode_categoricals(X):
    """Encode categorical columns."""
    for col in categorical_columns:
        X[col] = X[col].astype('category')
        X[col] = X[col].cat.codes
        print(f"{col} has {len(X[col].unique())} unique categories")
    return X

def analyze_features(X):
    """Analyze features and print correlations."""
    numerical_corr = X[numerical_columns].corr()
    print("\nFeature Correlations:")
    print(numerical_corr)
    return numerical_corr

def scale_features(X, is_training=True):
    """Scale numerical features."""
    if is_training:
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    else:
        X[numerical_columns] = scaler.transform(X[numerical_columns])
    return X

def preprocess_data(df):
    """Preprocess the input DataFrame."""
    # Print initial class distribution to verify balance
    print("\nClass distribution:")
    print(df[target_column].value_counts(normalize=True))

    # Drop excluded columns
    df = df.drop(columns=['Churn Reason', 'Churn Category', 'City', 'State', 'Zip Code', 'Latitude', 'Longitude'])

    # Split features and target
    X = df.drop(columns=[target_column, id_column])
    y = df[target_column]

    # Process features
    X = encode_categoricals(X)
    analyze_features(X)
    X = scale_features(X)

    # Split the data with stratification to maintain balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def get_model_params():
    """Get parameters for the model."""
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

# Training the model with GridSearchCV
def train(X_train, y_train):
    base_model = XGBClassifier(
        random_state=42,
        enable_categorical=True,
        tree_method='hist'
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=get_model_params(),
        cv=StratifiedKFold(n_splits=4),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Print feature importance
    print_feature_importance(model)
    return model

def print_feature_importance(model):
    """Print feature importance of the trained model."""
    if model is not None:
        feature_importance = pd.DataFrame({
            'feature': numerical_columns + categorical_columns,
            'importance': model.feature_importances_
        })
        print("\nFeature Importances:")
        sorted_importance = feature_importance.sort_values('importance', ascending=False)
        print(sorted_importance)
        return sorted_importance
    
def evaluate(model, X_test, y_test):
    """Evaluate the model using the test data."""
    if model is None:
        raise ValueError("Model hasn't been trained yet!")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

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

def save(model, model_dir='./model'):
    """Save the trained model and scaler."""
    if model is None:
        raise ValueError("No model to save!")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save components
    joblib.dump(model, f'{model_dir}/churn_model.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump((numerical_columns, categorical_columns), f'{model_dir}/columns.pkl')
    print(f"Model and components saved to {model_dir}")

def main():
    """Main function to run the predictor."""
    # Load data
    df = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train model
    model = train(X_train, y_train)

    # Evaluate model
    evaluate(model, X_test, y_test)

    # Save model
    save(model)

if __name__ == "__main__":
    main()