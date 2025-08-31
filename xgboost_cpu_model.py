import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

def train_model_cpu(X_train, y_train, X_test, y_test):
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
        num_boost_round=100,
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
    df = load_data()
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)
    print("\n--- Training with CPU ---")
    model_cpu, time_cpu = train_model_cpu(X_train, y_train, X_test, y_test)
    plot_feature_importance(model_cpu, feature_columns)
    model_cpu.save_model('stock_prediction_model_cpu.json')
    print("Model saved as 'stock_prediction_model_cpu.json'")

if __name__ == "__main__":
    main()
