import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path='combined_data/full_dataset.csv', sample_frac=1.0):
    """Load the dataset, optionally sampling a fraction of it"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    print(f"Loaded dataset with {len(df)} rows")
    return df

def prepare_data(df):
    """Prepare data for XGBoost training"""
    # Define features to use
    feature_columns = [col for col in df.columns if col not in [
        'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'Target_Return', 'Target_Direction', 'Date'
    ]]
    
    # Remove any remaining NaN values
    df = df.dropna(subset=feature_columns + ['Target_Direction'])
    
    # Split data
    X = df[feature_columns]
    y = df['Target_Direction']
    
    # Use 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {len(X_train)} rows")
    print(f"Testing set: {len(X_test)} rows")
    
    return X_train, X_test, y_train, y_test, feature_columns

def train_model_with_gpu(X_train, y_train, X_test, y_test, use_gpu=True):
    """Train XGBoost model with GPU acceleration if available"""
    # Set up parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'scale_pos_weight': 1,
    }
    
    if use_gpu:
        # Add GPU parameters
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
        print("Using GPU acceleration")
    else:
        params['tree_method'] = 'hist'
        print("Using CPU only")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train model and measure time
    start_time = time.time()
    
    # Train for 100 rounds
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(dtest)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return model, training_time

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    
    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Take top 20 features
    importance_df = importance_df.head(20)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

def main():
    # Load the dataset
    df = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)
    
    # Train with GPU
    print("\n--- Training with GPU ---")
    model_gpu, time_gpu = train_model_with_gpu(X_train, y_train, X_test, y_test, use_gpu=True)
    
    # Train with CPU for comparison
    print("\n--- Training with CPU ---")
    model_cpu, time_cpu = train_model_with_gpu(X_train, y_train, X_test, y_test, use_gpu=False)
    
    # Print speedup
    speedup = time_cpu / time_gpu
    print(f"\nGPU training was {speedup:.2f}x faster than CPU training")
    
    # Plot feature importance
    plot_feature_importance(model_gpu, feature_columns)
    
    # Save the model
    model_gpu.save_model('stock_prediction_model.json')
    print("Model saved as 'stock_prediction_model.json'")

if __name__ == "__main__":
    main()