import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
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

def tune_hyperparameters(X_train, y_train, use_gpu=True):
    """Tune hyperparameters for the XGBoost model"""
    print(f"Starting hyperparameter tuning for {'GPU' if use_gpu else 'CPU'} model...")
    start_time = time.time()
    
    # Clean data for hyperparameter tuning
    X_train_clean = X_train[np.isfinite(X_train.values).all(axis=1)]
    y_train_clean = y_train[np.isfinite(X_train.values).all(axis=1)]
    
    # Create parameter grid for tuning
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Create the XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='gpu_hist' if use_gpu else 'hist',
        gpu_id=0 if use_gpu else None,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Use F1 score as the scoring metric
    f1_scorer = make_scorer(f1_score)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=3,
        n_jobs=-1 if not use_gpu else 1,  # Use 1 job for GPU to avoid memory issues
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_clean, y_train_clean)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    end_time = time.time()
    print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds")
    
    # Return the best parameters
    return best_params

def train_model_with_gpu(X_train, y_train, X_test, y_test, use_gpu=True, best_params=None):
    """Train XGBoost model with GPU acceleration if available and with tuned parameters if provided"""
    # Set up parameters
    if best_params is None:
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
    else:
        # Convert GridSearchCV parameters to XGBoost params format
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': best_params.get('max_depth', 6),
            'eta': best_params.get('learning_rate', 0.1),
            'subsample': best_params.get('subsample', 0.8),
            'colsample_bytree': best_params.get('colsample_bytree', 0.8),
            'min_child_weight': best_params.get('min_child_weight', 1),
            'gamma': best_params.get('gamma', 0),
        }
        print(f"Using tuned parameters: {params}")
        print(f"Number of boosting rounds: {best_params.get('n_estimators', 100)}")
    
    if use_gpu:
        # Add GPU parameters
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
        print("Using GPU acceleration")
    else:
        params['tree_method'] = 'hist'
        print("Using CPU only")
    
    # Remove inf/-inf and very large values from X_train/X_test
    for df, name in [(X_train, 'X_train'), (X_test, 'X_test')]:
        mask = np.isfinite(df.values).all(axis=1)
        if not np.all(mask):
            print(f"Removed {np.sum(~mask)} rows with inf/-inf or NaN from {name}")
        df = df[mask]
        if name == 'X_train':
            y_train = y_train[mask]
        else:
            y_test = y_test[mask]
    # Reassign cleaned data
    X_train_clean = X_train[np.isfinite(X_train.values).all(axis=1)]
    y_train_clean = y_train
    X_test_clean = X_test[np.isfinite(X_test.values).all(axis=1)]
    y_test_clean = y_test

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_clean, label=y_train_clean)
    dtest = xgb.DMatrix(X_test_clean, label=y_test_clean)
    
    # Train model and measure time
    start_time = time.time()
    
    # Train with the specified number of rounds or default to 100
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_params.get('n_estimators', 100) if best_params else 100,
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
    
    # Tune hyperparameters for GPU
    print("\n--- Hyperparameter Tuning for GPU Model ---")
    best_params_gpu = tune_hyperparameters(X_train, y_train, use_gpu=True)
    
    # Tune hyperparameters for CPU (for comparison)
    print("\n--- Hyperparameter Tuning for CPU Model ---")
    best_params_cpu = tune_hyperparameters(X_train, y_train, use_gpu=False)
    
    # Train with GPU using tuned parameters
    print("\n--- Training with GPU using tuned parameters ---")
    model_gpu, time_gpu = train_model_with_gpu(X_train, y_train, X_test, y_test, use_gpu=True, best_params=best_params_gpu)
    
    # Train with CPU for comparison using tuned parameters
    print("\n--- Training with CPU using tuned parameters ---")
    model_cpu, time_cpu = train_model_with_gpu(X_train, y_train, X_test, y_test, use_gpu=False, best_params=best_params_cpu)
    
    # Print speedup
    speedup = time_cpu / time_gpu
    print(f"\nGPU training was {speedup:.2f}x faster than CPU training")
    
    # Plot feature importance
    plot_feature_importance(model_gpu, feature_columns)
    
    # Save the models
    model_gpu.save_model('stock_prediction_model_gpu.json')
    print("GPU model saved as 'stock_prediction_model_gpu.json'")
    
    model_cpu.save_model('stock_prediction_model_cpu_from_gpu_script.json')
    print("CPU model saved as 'stock_prediction_model_cpu_from_gpu_script.json'")
    
    # Save best parameters to files for future reference
    with open('best_params_gpu.txt', 'w') as f:
        f.write(str(best_params_gpu))
    print("Best GPU parameters saved to 'best_params_gpu.txt'")
    
    with open('best_params_cpu_from_gpu_script.txt', 'w') as f:
        f.write(str(best_params_cpu))
    print("Best CPU parameters saved to 'best_params_cpu_from_gpu_script.txt'")

if __name__ == "__main__":
    main()