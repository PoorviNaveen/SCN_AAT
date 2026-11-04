"""
Data Preprocessing for ANFIS vs ANFIS-GA Traffic Flow Prediction
Paper: "Comparative Study Analysis of ANFIS and ANFIS-GA Models on Flow of Vehicles at Road Intersections"
Dataset: Metro Interstate Traffic Volume (UCI)

NOTE: The paper used 850 traffic samples (Table 4).
      We sample 850 rows from the UCI dataset to exactly match the paper's scale.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Metro_Interstate_Traffic_Volume.csv')


def load_and_preprocess(n_samples=850, random_state=42, include_weekend=True, train_split=0.70):
    """
    Load and preprocess the dataset.
    Features:
      - Input 1 : time (hour of day)
      - Input 2 : temp_celsius (environmental weather condition)
      - Input 3 : is_weekend (binary commuter vs weekend pattern) [Enhanced feature]
      - Output  : traffic_volume (normalised)
    Parameters:
      - n_samples: Total sample count (850 matches paper Table 4 scale)
      - train_split: Train partition ratio (0.70 for 70:30, 0.80 for 80:20)
    """
    df = pd.read_csv(DATA_PATH, parse_dates=['date_time'])

    # Feature Engineering
    df['hour']         = df['date_time'].dt.hour
    df['temp_celsius'] = df['temp'] - 273.15
    df['dayofweek']    = df['date_time'].dt.dayofweek
    df['is_weekend']   = (df['dayofweek'] >= 5).astype(float)

    df = df.dropna(subset=['temp', 'traffic_volume'])

    # Sample 850 rows to match paper scale (paper: 850 samples, Table 4)
    df_sample = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    if include_weekend:
        feature_cols = ['hour', 'temp_celsius', 'is_weekend']
    else:
        feature_cols = ['hour', 'temp_celsius']

    X = df_sample[feature_cols].values.astype(float)
    y = df_sample['traffic_volume'].values.astype(float).reshape(-1, 1)

    # Train / Test split BEFORE scaling (Leak-free: test set statistics do not leak into train)
    split    = int(train_split * len(df_sample))
    X_train_raw = X[:split]
    X_test_raw  = X[split:]
    y_train_raw = y[:split]
    y_test_raw  = y[split:]

    # Fit scalers STRICTLY on training split
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train  = scaler_X.fit_transform(X_train_raw)
    X_test   = scaler_X.transform(X_test_raw)
    y_train  = scaler_y.fit_transform(y_train_raw).ravel()
    y_test   = scaler_y.transform(y_test_raw).ravel()

    train_pct = int(round(train_split * 100))
    test_pct  = 100 - train_pct

    print("=" * 55)
    print(f"  Dataset Summary ({train_pct}:{test_pct} Split)")
    print("=" * 55)
    print(f"  Total samples      : {len(df_sample)}  (paper scale: 850)")
    print(f"  Training ({train_pct}%)     : {len(X_train)}")
    print(f"  Testing  ({test_pct}%)     : {len(X_test)}")
    print(f"  Features ({len(feature_cols)})        : {', '.join(feature_cols)}")
    print(f"  Output             : traffic_volume (normalised)")
    print("=" * 55)
    sys.stdout.flush()

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, df_sample


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, df = load_and_preprocess()
    print("\nFirst 5 training samples:")
    print(pd.DataFrame(X_train[:5], columns=['hour_norm', 'temp_norm']))
