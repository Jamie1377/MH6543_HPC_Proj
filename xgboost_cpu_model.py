import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import time
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_data(file_path='combined_data/full_dataset.csv', sample_frac=1.0):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    print(f"Loaded dataset with {len(df)} rows")
    return df

def prepare_data(df):
    feature_columns = [col for col in df.columns if col not in [
        'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'Target_Return', 'Target_Direction', 'Date']]
    df = df.dropna(subset=feature_columns + ['Target_Direction'])
    X = df[feature_columns]
    y = df['Target_Direction']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    print(f"Training set: {len(X_train)} rows")
    print(f"Testing set: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test, feature_columns

def tune_hyperparameters_cpu(X_train, y_train, search_method = 'random'):

    # Ask user for tuning method preference
    

    
    print(f"Starting hyperparameter tuning with {'RandomizedSearchCV' if search_method == 'random' else 'GridSearchCV'}...")
    start_time = time.time()
    
    # Clean data for hyperparameter tuning
    X_train_clean = X_train[np.isfinite(X_train.values).all(axis=1)]
    y_train_clean = y_train[np.isfinite(X_train.values).all(axis=1)]
    
    # Remove rows with inf/-inf or NaN from X_train
    mask = np.isfinite(X_train.values).all(axis=1)
    if not np.all(mask):
        print(f"Removed {np.sum(~mask)} rows with inf/-inf or NaN from X_train")
    
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
    
    # Create the XGBoost classifier for CPU
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Use F1 score as the scoring metric
    f1_scorer = make_scorer(f1_score)
    
    # Set up appropriate search strategy based on user preference
    if search_method == 'random':
        # Set up RandomizedSearchCV with 20 iterations (much faster than GridSearchCV)
        search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # Try 20 parameter combinations
            scoring=f1_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # Set up GridSearchCV (more thorough but much slower)
        search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
    
    # Fit the search
    search.fit(X_train_clean, y_train_clean)
    
    # Get the best parameters
    best_params = search.best_params_
    print(f"Best parameters found: {best_params}")
    print(f"Best F1 score: {search.best_score_:.4f}")
    
    end_time = time.time()
    print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds")
    
    # Return the best parameters
    return best_params

def train_model_cpu(X_train, y_train, X_test, y_test, best_params=None):
    # Default parameters if no tuned parameters are provided
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
            'tree_method': 'hist'
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
            'tree_method': 'hist'
        }
        
        # Use the tuned number of boosting rounds
        num_boost_round = best_params.get('n_estimators', 100)
        print(f"Using tuned parameters: {params}")
        print(f"Number of boosting rounds: {num_boost_round}")
    # Remove inf/-inf and NaN from X_train/X_test
    for df, name in [(X_train, 'X_train'), (X_test, 'X_test')]:
        mask = np.isfinite(df.values).all(axis=1)
        if not np.all(mask):
            print(f"Removed {np.sum(~mask)} rows with inf/-inf or NaN from {name}")
        df = df[mask]
        if name == 'X_train':
            y_train = y_train[mask]
        else:
            y_test = y_test[mask]
    X_train_clean = X_train[np.isfinite(X_train.values).all(axis=1)]
    y_train_clean = y_train
    X_test_clean = X_test[np.isfinite(X_test.values).all(axis=1)]
    y_test_clean = y_test
    dtrain = xgb.DMatrix(X_train_clean, label=y_train_clean)
    dtest = xgb.DMatrix(X_test_clean, label=y_test_clean)
    start_time = time.time()
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
    y_pred = model.predict(dtest)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    accuracy = accuracy_score(y_test_clean, y_pred_binary)
    precision = precision_score(y_test_clean, y_pred_binary)
    recall = recall_score(y_test_clean, y_pred_binary)
    f1 = f1_score(y_test_clean, y_pred_binary)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return model, training_time

def plot_feature_importance(model, feature_names):
    importance = model.get_score(importance_type='gain')
    importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    importance_df = importance_df.head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_cpu.png')
    print("Feature importance plot saved as 'feature_importance_cpu.png'")

def main():
    time_start = time.time()
    df = load_data()
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)
    
    # Perform hyperparameter tuning
    print("\n--- Hyperparameter Tuning for CPU Model ---")
    best_params = tune_hyperparameters_cpu(X_train, y_train)
    
    # Train with tuned parameters
    print("\n--- Training with CPU using tuned parameters ---")
    model_cpu, time_cpu = train_model_cpu(X_train, y_train, X_test, y_test, best_params=best_params)
    
    plot_feature_importance(model_cpu, feature_columns)
    model_cpu.save_model('stock_prediction_model_cpu.json')
    print("Model saved as 'stock_prediction_model_cpu.json'")
    
    time_end = time.time()
    print(f"\nTotal execution time: {time_end - time_start:.2f} seconds")
    
    # Save best parameters to a file for future reference
    with open('best_params_cpu.txt', 'w') as f:
        f.write(str(best_params))
    print("Best parameters saved to 'best_params_cpu.txt'")
if __name__ == "__main__":
    main()
