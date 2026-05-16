import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_train():
    data_path = r"C:\Users\HIMASHEKAR\OneDrive\Desktop\parkinson_disease.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load data
    df_raw = pd.read_csv(data_path)
    print("Dataset loaded successfully.")

    # -------------------------------------------------------------
    # 1. GFG PRIMARY PIPELINE (For Colab Match)
    # -------------------------------------------------------------
    df = df_raw.copy()
    
    # Preprocessing (GFG style)
    if 'id' in df.columns:
        df = df.groupby('id').mean().reset_index()
        df.drop('id', axis=1, inplace=True)
        
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
        
    # Correlation Filtering (> 0.7)
    corr_matrix = df.drop('class', axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    df_filtered = df.drop(columns=to_drop)

    X = df_filtered.drop('class', axis=1)
    y = df_filtered['class']

    # SelectKBest (chi2, k=30)
    scaler_gfg = MinMaxScaler()
    X_norm = scaler_gfg.fit_transform(X)
    
    # If the filtered features are less than 30, adjust k
    k_val = min(30, X.shape[1])
    selector = SelectKBest(chi2, k=k_val)
    selector.fit(X_norm, y)
    
    selected_cols = X.columns[selector.get_support()]
    X_selected = X[selected_cols]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=10)
    
    # Better approach for inference: Save standard scaler fit on X_selected
    inf_scaler = MinMaxScaler()
    X_train_scaled = inf_scaler.fit_transform(X_train)
    X_test_scaled = inf_scaler.transform(X_test)
    
    # Train Logistic Regression (GFG Selected Model)
    print("Training GFG Primary Model (Logistic Regression)...")
    # We remove RandomOverSampler so the model naturally heavily favors "Healthy", matching real-world distributions.
    log_model = LogisticRegression()
    log_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_preds = log_model.predict(X_train_scaled)
    test_preds = log_model.predict(X_test_scaled)
    
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    cm = confusion_matrix(y_test, test_preds)
    
    print(f"GFG Model Train AUC: {train_auc:.4f}")
    print(f"GFG Model Test AUC: {test_auc:.4f}")

    # Save GFG artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump(log_model, f)
    with open('scaler.pkl', 'wb') as f:
        # We need a scaler for inference, we'll save the selector too or just use the filtered columns
        pass # We will save a combined pipeline or standard scaler below
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(inf_scaler, f)
        
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(selected_cols.tolist(), f)
    with open('background_data.pkl', 'wb') as f:
        # Save background data for SHAP Explainer
        pickle.dump(X_train_scaled, f)

    # -------------------------------------------------------------
    # 2. LIVE AUDIO PIPELINE (MFCC Only)
    # -------------------------------------------------------------
    df_mfcc = df_raw.copy()
    if 'id' in df_mfcc.columns:
        df_mfcc = df_mfcc.drop(columns=['id'])
    if 'name' in df_mfcc.columns:
        df_mfcc = df_mfcc.drop(columns=['name'])
        
    X_all = df_mfcc.drop(columns=['class'])
    y_all = df_mfcc['class']
    
    # We drop the 0th coefficient (energy) to make the model robust to microphone volume differences.
    mfcc_features = [
        'mean_MFCC_1st_coef', 'mean_MFCC_2nd_coef', 'mean_MFCC_3rd_coef', 
        'mean_MFCC_4th_coef', 'mean_MFCC_5th_coef', 'mean_MFCC_6th_coef', 'mean_MFCC_7th_coef', 
        'mean_MFCC_8th_coef', 'mean_MFCC_9th_coef', 'mean_MFCC_10th_coef', 'mean_MFCC_11th_coef', 
        'mean_MFCC_12th_coef'
    ]
    
    X_mfcc = X_all[mfcc_features]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mfcc, y_all, test_size=0.2, random_state=42, stratify=y_all)
    
    scaler_mfcc = StandardScaler()
    X_train_m_scaled = scaler_mfcc.fit_transform(X_train_m)
    X_test_m_scaled = scaler_mfcc.transform(X_test_m)
    
    xgb_model_mfcc = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42, 
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )
    print("Training MFCC Model (For Live Audio)...")
    xgb_model_mfcc.fit(X_train_m_scaled, y_train_m)
    
    y_pred_m = xgb_model_mfcc.predict(X_test_m_scaled)
    acc_m = accuracy_score(y_test_m, y_pred_m)
    
    with open('model_mfcc.pkl', 'wb') as f:
        pickle.dump(xgb_model_mfcc, f)
    with open('scaler_mfcc.pkl', 'wb') as f:
        pickle.dump(scaler_mfcc, f)
    with open('feature_names_mfcc.pkl', 'wb') as f:
        pickle.dump(mfcc_features, f)

    # Save metrics
    metrics = {
        'gfg_train_auc': float(train_auc),
        'gfg_test_auc': float(test_auc),
        'gfg_cm': cm.tolist(),
        'mfcc_accuracy': float(acc_m),
        'selected_features': selected_cols.tolist()
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("\nAll models and scalers saved successfully.")

if __name__ == "__main__":
    load_and_train()
