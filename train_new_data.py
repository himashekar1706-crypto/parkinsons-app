import pandas as pd
import numpy as np
import librosa
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def extract_features_from_folder(folder_path, label):
    features_list = []
    
    # We expect 12 MFCC means (dropping the 0th coefficient)
    mfcc_columns = [
        'mean_MFCC_1st_coef', 'mean_MFCC_2nd_coef', 'mean_MFCC_3rd_coef', 
        'mean_MFCC_4th_coef', 'mean_MFCC_5th_coef', 'mean_MFCC_6th_coef', 'mean_MFCC_7th_coef', 
        'mean_MFCC_8th_coef', 'mean_MFCC_9th_coef', 'mean_MFCC_10th_coef', 'mean_MFCC_11th_coef', 
        'mean_MFCC_12th_coef'
    ]
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_means = np.mean(mfccs, axis=1)
                
                # Drop 0th coefficient
                features = mfcc_means[1:]
                
                # Append to list
                row_data = list(features) + [label]
                features_list.append(row_data)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    df = pd.DataFrame(features_list, columns=mfcc_columns + ['class'])
    return df

def retrain_model():
    print("Extracting features from new data...")
    hc_path = r"C:\Users\HIMASHEKAR\Downloads\New folder\HC_AH\HC_AH"
    pd_path = r"C:\Users\HIMASHEKAR\Downloads\New folder\PD_AH\PD_AH"
    
    df_hc = extract_features_from_folder(hc_path, 0)
    df_pd = extract_features_from_folder(pd_path, 1)
    
    df_new = pd.concat([df_hc, df_pd], ignore_index=True)
    print(f"Extracted features from {len(df_new)} new audio files.")
    
    # Load old data
    print("Loading original dataset...")
    df_old_raw = pd.read_csv(r"C:\Users\HIMASHEKAR\OneDrive\Desktop\parkinson_disease.csv")
    if 'id' in df_old_raw.columns:
        df_old_raw = df_old_raw.drop(columns=['id'])
    if 'name' in df_old_raw.columns:
        df_old_raw = df_old_raw.drop(columns=['name'])
        
    mfcc_features = [
        'mean_MFCC_1st_coef', 'mean_MFCC_2nd_coef', 'mean_MFCC_3rd_coef', 
        'mean_MFCC_4th_coef', 'mean_MFCC_5th_coef', 'mean_MFCC_6th_coef', 'mean_MFCC_7th_coef', 
        'mean_MFCC_8th_coef', 'mean_MFCC_9th_coef', 'mean_MFCC_10th_coef', 'mean_MFCC_11th_coef', 
        'mean_MFCC_12th_coef'
    ]
    
    df_old = df_old_raw[mfcc_features + ['class']]
    
    # Combine datasets
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    print(f"Combined dataset shape: {df_combined.shape}")
    
    X = df_combined[mfcc_features]
    y = df_combined['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler_mfcc = StandardScaler()
    X_train_scaled = scaler_mfcc.fit_transform(X_train)
    X_test_scaled = scaler_mfcc.transform(X_test)
    
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
    
    print("Retraining XGBoost model on combined data...")
    xgb_model_mfcc.fit(X_train_scaled, y_train)
    
    y_pred = xgb_model_mfcc.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"New Model Accuracy: {acc*100:.2f}%")
    
    # Save artifacts
    with open('model_mfcc.pkl', 'wb') as f:
        pickle.dump(xgb_model_mfcc, f)
    with open('scaler_mfcc.pkl', 'wb') as f:
        pickle.dump(scaler_mfcc, f)
        
    print("Updated model and scaler saved.")
    
    # Update metrics.json
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
        
    metrics['mfcc_accuracy'] = float(acc)
    metrics['dataset_size'] = len(df_combined)
    metrics['new_data_added'] = len(df_new)
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("Updated metrics.json.")

if __name__ == "__main__":
    retrain_model()
