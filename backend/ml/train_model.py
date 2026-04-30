"""
DialysisGuard Model Training Pipeline (v2)

Trains an Attention-Augmented BiGRU model on synthetic_hemodialysis_timeseries.csv
for hemodialysis instability prediction.

Key changes from v1:
    - Per-SEQUENCE classification (uses last-timestep Is_Unstable as label)
    - Sliding window augmentation (trains on sub-sequences of varying lengths)
    - Scaler fitted on TRAINING SET only (no data snooping)
    - Separated static/temporal feature processing
    - BiGRU architecture (bidirectional)
    - Class imbalance handling via weighted loss (17.6% positive rate at row level)

Usage:
    python train_model.py                  # Train and save model
    python train_model.py --validate       # Train, validate, and print metrics

Output artifacts (saved to backend/ml/):
    - gru_model.h5           : Trained model weights
    - scaler.pkl             : Fitted StandardScaler (train-only)
    - label_encoders.pkl     : Fitted LabelEncoders for categorical features
    - feature_config.json    : Feature names, types, clinical ranges
    - model_card.json        : Model transparency card with metrics
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

# Suppress TF warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
sys.setrecursionlimit(10000)  # Needed for Keras model serialization with custom layers

import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_gru import AttentionGRUModel, BahdanauAttention

# ============================================================
# Configuration
# ============================================================

# Path to training data
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'synthetic_hemodialysis_timeseries.csv'
)

# Output directory for saved artifacts
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Feature definitions
STATIC_FEATURES = [
    'Age', 'Gender', 'Weight', 'Diabetes', 'Hypertension',
    'Kidney Failure Cause', 'Creatinine', 'Urea', 'Potassium',
    'Hemoglobin', 'Hematocrit', 'Albumin', 'Dialysis Duration (hours)',
    'Dialysis Frequency (per week)', 'Dialysate Composition',
    'Vascular Access Type', 'Dialyzer Type', 'Urine Output (ml/day)',
    'Dry Weight (kg)', 'Fluid Removal Rate (ml/hour)', 'Disease Severity'
]

TEMPORAL_FEATURES = ['Current_BP', 'Current_HR', 'Time_Minutes']

CATEGORICAL_FEATURES = [
    'Gender', 'Kidney Failure Cause', 'Dialysate Composition',
    'Vascular Access Type', 'Dialyzer Type', 'Disease Severity'
]

BOOLEAN_FEATURES = ['Diabetes', 'Hypertension']

TARGET = 'Is_Unstable'

# Model hyperparameters
SEQUENCE_LENGTH = 30  # 30 time steps per patient
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15

# Sliding window sizes for augmentation
WINDOW_SIZES = [10, 15, 20, 25, 30]


# ============================================================
# Data Loading & Preprocessing
# ============================================================

def load_data(data_path):
    """Load and validate the training dataset."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"  Shape: {df.shape}")
    print(f"  Patients: {df['Patient_ID'].nunique()}")
    print(f"  Time steps per patient: {df.groupby('Patient_ID').size().unique()}")
    print(f"  Target distribution (row-level): {dict(df[TARGET].value_counts())}")
    
    # Show per-timestep variability
    vary = df.groupby('Patient_ID')[TARGET].nunique()
    print(f"  Patients with varying Is_Unstable: {(vary > 1).sum()}")
    print(f"  Patients always stable: {(vary == 1).sum() - ((vary == 1) & (df.groupby('Patient_ID')[TARGET].first() == 1)).sum()}")
    
    return df


def engineer_features(df):
    """
    Create derived features that capture temporal dynamics.
    These help the GRU model detect deterioration patterns.
    """
    print("Engineering temporal features...")
    
    # Sort by patient and time
    df = df.sort_values(['Patient_ID', 'Time_Minutes']).reset_index(drop=True)
    
    # BP rate of change (within each patient)
    df['BP_Change'] = df.groupby('Patient_ID')['Current_BP'].diff().fillna(0)
    
    # HR rate of change
    df['HR_Change'] = df.groupby('Patient_ID')['Current_HR'].diff().fillna(0)
    
    # BP deviation from patient's initial BP (first reading)
    initial_bp = df.groupby('Patient_ID')['Current_BP'].transform('first')
    df['BP_Deviation'] = df['Current_BP'] - initial_bp
    
    # HR deviation from patient's initial HR
    initial_hr = df.groupby('Patient_ID')['Current_HR'].transform('first')
    df['HR_Deviation'] = df['Current_HR'] - initial_hr
    
    # Cumulative BP volatility (rolling std over last 5 steps)
    df['BP_Volatility'] = df.groupby('Patient_ID')['Current_BP'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    ).fillna(0)
    
    # HR volatility
    df['HR_Volatility'] = df.groupby('Patient_ID')['Current_HR'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    ).fillna(0)
    
    # Risk score: composite of fluid removal rate and BP drop
    fluid_rate = df['Fluid Removal Rate (ml/hour)']
    weight = df['Weight']
    df['Fluid_Rate_Per_Kg'] = fluid_rate / weight
    
    # Session progress (0 to 1)
    df['Session_Progress'] = df['Time_Minutes'] / 232.0
    
    print(f"  Added 8 engineered features")
    
    return df


def encode_features(df, fit=True, label_encoders=None):
    """Encode categorical and boolean features."""
    print("Encoding features...")
    
    if label_encoders is None:
        label_encoders = {}
    
    # Encode boolean features as int
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            df[col] = label_encoders[col].transform(df[col].astype(str))
    
    print(f"  Encoded {len(BOOLEAN_FEATURES)} boolean + {len(CATEGORICAL_FEATURES)} categorical features")
    
    return df, label_encoders


def prepare_sequences(df, scaler=None, fit_scaler=True):
    """
    Prepare sequences for the BiGRU model.
    
    Each patient's 30 time-step record becomes one sequence.
    The label for each sequence is the Is_Unstable value at the LAST timestep.
    
    Returns:
        X: (n_patients, 30, n_features)
        y: (n_patients,) - Is_Unstable at last timestep of each sequence
        y_per_step: (n_patients, 30) - per-timestep labels for sliding window
        feature_names: list of feature names
    """
    print("Preparing sequences...")
    
    # All features to use (static + temporal + engineered)
    engineered_features = [
        'BP_Change', 'HR_Change', 'BP_Deviation', 'HR_Deviation',
        'BP_Volatility', 'HR_Volatility', 'Fluid_Rate_Per_Kg',
        'Session_Progress'
    ]
    
    # IMPORTANT: Order matters — static first, then temporal+engineered
    # This matches the model's feature slicing in attention_gru.py
    all_features = STATIC_FEATURES + TEMPORAL_FEATURES + engineered_features
    feature_names = all_features.copy()
    
    # Sort and group by patient
    df = df.sort_values(['Patient_ID', 'Time_Minutes'])
    patients = df['Patient_ID'].unique()
    
    X_list = []
    y_list = []
    y_per_step_list = []
    
    for pid in patients:
        patient_data = df[df['Patient_ID'] == pid]
        
        # Get feature matrix for this patient (30 × n_features)
        X_patient = patient_data[all_features].values
        
        # Target: Is_Unstable at the LAST timestep of the sequence
        y_patient = int(patient_data[TARGET].iloc[-1])
        
        # Per-timestep labels (for sliding window augmentation)
        y_steps = patient_data[TARGET].values.astype(np.float32)
        
        X_list.append(X_patient)
        y_list.append(y_patient)
        y_per_step_list.append(y_steps)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    y_per_step = np.array(y_per_step_list, dtype=np.float32)
    
    print(f"  Sequence shape: {X.shape}")
    print(f"  Target distribution (last-step): stable={int((y == 0).sum())}, unstable={int((y == 1).sum())}")
    
    return X, y, y_per_step, feature_names


def create_sliding_windows(X, y_per_step, min_window=10):
    """
    Create sliding window subsequences for augmentation.
    
    This teaches the model to predict from partial data (10-30 steps),
    fixing the zero-padding OOD problem at inference time.
    
    Args:
        X: Sequences (n_patients, 30, n_features) — already split (train only)
        y_per_step: Per-timestep labels (n_patients, 30) — matching X
        min_window: Minimum window size
    
    Returns:
        X_aug: Augmented sequences (padded to 30 steps)
        y_aug: Labels for each window (Is_Unstable at last step of window)
    """
    print("Creating sliding window augmentations...")
    
    X_aug = []
    y_aug = []
    n_patients, n_steps, n_features = X.shape
    
    for i in range(n_patients):
        for window_size in WINDOW_SIZES:
            if window_size > n_steps:
                continue
            
            # Take the last `window_size` steps
            X_window = X[i, -window_size:, :]
            y_window = float(y_per_step[i, -1])  # Last-step label
            
            # Pad at the beginning with the first step's values (not zeros!)
            # This avoids OOD zero-padding
            if window_size < n_steps:
                pad_length = n_steps - window_size
                first_step = X_window[0:1, :]  # (1, n_features)
                padding = np.repeat(first_step, pad_length, axis=0)
                X_padded = np.concatenate([padding, X_window], axis=0)
            else:
                X_padded = X_window
            
            X_aug.append(X_padded)
            y_aug.append(y_window)
            
            # Also create a window from the middle for more diversity
            if window_size < n_steps and window_size >= min_window:
                start = max(0, (n_steps - window_size) // 2)
                X_mid = X[i, start:start + window_size, :]
                y_mid = float(y_per_step[i, start + window_size - 1])
                
                pad_length = n_steps - window_size
                first_step = X_mid[0:1, :]
                padding = np.repeat(first_step, pad_length, axis=0)
                X_padded_mid = np.concatenate([padding, X_mid], axis=0)
                
                X_aug.append(X_padded_mid)
                y_aug.append(y_mid)
    
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.float32)
    
    print(f"  Augmented dataset: {X_aug.shape[0]} sequences (from {n_patients} patients)")
    print(f"  Augmented target: stable={int((y_aug == 0).sum())}, unstable={int((y_aug == 1).sum())}")
    
    return X_aug, y_aug


def scale_data(X_train, X_val, X_test, X_aug=None):
    """
    Scale features using StandardScaler fitted on TRAINING DATA ONLY.
    
    Returns scaled arrays and fitted scaler.
    """
    print("Scaling features (train-only fit)...")
    
    n_patients_train, n_timesteps, n_features = X_train.shape
    
    # Fit scaler on training data only
    X_train_flat = X_train.reshape(-1, n_features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train = X_train_scaled.reshape(n_patients_train, n_timesteps, n_features).astype(np.float32)
    
    # Transform val and test
    X_val_flat = X_val.reshape(-1, n_features)
    X_val = scaler.transform(X_val_flat).reshape(X_val.shape).astype(np.float32)
    
    X_test_flat = X_test.reshape(-1, n_features)
    X_test = scaler.transform(X_test_flat).reshape(X_test.shape).astype(np.float32)
    
    # Transform augmented data if provided
    if X_aug is not None:
        X_aug_flat = X_aug.reshape(-1, n_features)
        X_aug = scaler.transform(X_aug_flat).reshape(X_aug.shape).astype(np.float32)
    
    # Handle any NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    if X_aug is not None:
        X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Scaler fitted on {n_patients_train * n_timesteps} training rows")
    
    return X_train, X_val, X_test, X_aug, scaler


def split_data(X, y, y_per_step=None, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=42):
    """Split data by patient (no data leakage). Also splits y_per_step if provided."""
    print("Splitting data...")
    
    # First split: train+val vs test
    if y_per_step is not None:
        X_trainval, X_test, y_trainval, y_test, yps_trainval, yps_test = train_test_split(
            X, y, y_per_step, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    if y_per_step is not None:
        X_train, X_val, y_train, y_val, yps_train, yps_val = train_test_split(
            X_trainval, y_trainval, yps_trainval, test_size=val_ratio,
            random_state=random_state, stratify=y_trainval
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio,
            random_state=random_state, stratify=y_trainval
        )
    
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    if y_per_step is not None:
        return X_train, X_val, X_test, y_train, y_val, y_test, yps_train
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Training
# ============================================================

def train_model(X_train, y_train, X_val, y_val, n_features,
                n_static=21, n_temporal=11,
                epochs=None, batch_size=None):
    """Build and train the Attention-Augmented BiGRU model."""
    _epochs = epochs or EPOCHS
    _batch_size = batch_size or BATCH_SIZE
    
    print("\n" + "=" * 60)
    print("BUILDING MODEL (v2 — BiGRU with separated features)")
    print("=" * 60)
    
    builder = AttentionGRUModel(
        n_timesteps=SEQUENCE_LENGTH,
        n_features=n_features,
        n_static=n_static,
        n_temporal=n_temporal,
        gru_units=(64, 32),
        attention_units=64,
        dense_units=(32,),
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    model = builder.build()
    builder.summary()
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Class weights to handle imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos > 0:
        weight_ratio = n_neg / n_pos
        # Cap at reasonable range
        weight_ratio = min(weight_ratio, 10.0)
    else:
        weight_ratio = 1.0
    class_weight = {0: 1.0, 1: weight_ratio}
    print(f"  Class weights: {class_weight}")
    print(f"  Positive rate: {n_pos / len(y_train):.2%}")
    
    # Checkpoint path for best model weights
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_weights.weights.h5')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=_epochs,
        batch_size=_batch_size,
        class_weight=class_weight,
        callbacks=builder.get_callbacks(
            patience=PATIENCE,
            checkpoint_path=checkpoint_path
        ),
        verbose=1
    )
    
    # Load best weights from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\n  Loading best weights from checkpoint: {checkpoint_path}")
        builder.model.load_weights(checkpoint_path)
    
    return builder, history


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(builder, X_test, y_test, feature_names):
    """Comprehensive model evaluation."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    model = builder.model
    
    # Standard predictions
    y_pred_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    {cm}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stable', 'Unstable']))
    
    # Test MC Dropout uncertainty
    print("\n  Testing MC Dropout uncertainty (20 passes)...")
    sample = X_test[:5]
    mc_results = builder.predict_with_dropout(sample, n_passes=20)
    means = mc_results['mean'] if isinstance(mc_results['mean'], list) else [mc_results['mean']]
    stds = mc_results['std'] if isinstance(mc_results['std'], list) else [mc_results['std']]
    print(f"    Mean predictions: {[f'{v:.3f}' for v in means]}")
    print(f"    Std deviations:   {[f'{v:.3f}' for v in stds]}")
    
    # Test attention weights
    print("\n  Testing attention weight extraction...")
    attn_weights = builder.get_attention_weights(sample)
    print(f"    Attention shape: {attn_weights.shape}")
    attn_sums = attn_weights.squeeze().sum(axis=-1)
    if attn_sums.ndim == 0:
        print(f"    Weights sum: {attn_sums:.4f}")
    else:
        print(f"    Weights sum per sample: {[f'{w:.4f}' for w in attn_sums[:3]]}")
    
    metrics = {
        'accuracy': round(float(accuracy), 4),
        'auc': round(float(auc), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


# ============================================================
# Save Artifacts
# ============================================================

def save_artifacts(builder, scaler, label_encoders, feature_names,
                   metrics, df, output_dir):
    """Save all training artifacts to disk."""
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save model weights (weights-only to avoid serialization recursion)
    weights_path = os.path.join(output_dir, 'model_weights.weights.h5')
    builder.model.save_weights(weights_path)
    print(f"  Weights saved: {weights_path}")
    
    # Save model architecture config for reconstruction
    arch_config = {
        'n_timesteps': builder.n_timesteps,
        'n_features': builder.n_features,
        'n_static': builder.n_static,
        'n_temporal': builder.n_temporal,
        'gru_units': list(builder.gru_units),
        'attention_units': builder.attention_units,
        'dense_units': list(builder.dense_units),
        'dropout_rate': builder.dropout_rate,
        'learning_rate': builder.learning_rate
    }
    arch_path = os.path.join(output_dir, 'model_architecture.json')
    with open(arch_path, 'w') as f:
        json.dump(arch_config, f, indent=2)
    print(f"  Architecture config saved: {arch_path}")
    
    # 2. Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")
    
    # 3. Save label encoders
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"  Label encoders saved: {encoders_path}")
    
    # 4. Save feature config
    n_static = len(STATIC_FEATURES)
    n_temporal = len(TEMPORAL_FEATURES)
    engineered = [
        'BP_Change', 'HR_Change', 'BP_Deviation', 'HR_Deviation',
        'BP_Volatility', 'HR_Volatility', 'Fluid_Rate_Per_Kg',
        'Session_Progress'
    ]
    n_engineered = len(engineered)
    
    feature_config = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_timesteps': SEQUENCE_LENGTH,
        'n_static': n_static,
        'n_temporal': n_temporal + n_engineered,  # temporal + engineered go through GRU
        'static_features': STATIC_FEATURES,
        'temporal_features': TEMPORAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'boolean_features': BOOLEAN_FEATURES,
        'engineered_features': engineered,
        'feature_ranges': {}
    }
    
    # Compute clinical ranges from training data
    for feat in feature_names:
        if feat in df.columns:
            feature_config['feature_ranges'][feat] = {
                'min': round(float(df[feat].min()), 2),
                'max': round(float(df[feat].max()), 2),
                'mean': round(float(df[feat].mean()), 2),
                'std': round(float(df[feat].std()), 2)
            }
    
    config_path = os.path.join(output_dir, 'feature_config.json')
    with open(config_path, 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"  Feature config saved: {config_path}")
    
    # 5. Save model card
    n_patients = df['Patient_ID'].nunique()
    
    # Row-level distribution
    row_dist = dict(df[TARGET].value_counts())
    # Per-patient last-step distribution
    last_step = df.groupby('Patient_ID').last()
    patient_dist = dict(last_step[TARGET].value_counts())
    
    model_card = {
        'model_name': 'DialysisGuard Attention-BiGRU v2.0',
        'model_type': 'Attention-Augmented BiGRU (64→32→Attention→Dense32)',
        'task': 'Binary classification — Hemodialysis instability prediction',
        'version': '2.0',
        'changes_from_v1': [
            'BiGRU (bidirectional) instead of unidirectional GRU',
            'Separated static/temporal feature processing',
            'Removed RepeatVector bottleneck',
            'Per-sequence classification using last-timestep label',
            'Sliding window augmentation for partial-sequence robustness',
            'Scaler fitted on training data only (no data snooping)'
        ],
        'training_dataset': {
            'source': 'synthetic_hemodialysis_timeseries.csv',
            'total_rows': len(df),
            'patients': n_patients,
            'time_steps_per_patient': SEQUENCE_LENGTH,
            'features': len(feature_names),
            'n_static_features': n_static,
            'n_temporal_features': n_temporal + n_engineered,
            'target_distribution_row_level': {
                'stable': int(row_dist.get(0, 0)),
                'unstable': int(row_dist.get(1, 0)),
                'positive_rate': f"{int(row_dist.get(1, 0)) / len(df):.1%}"
            },
            'target_distribution_last_step': {
                'stable': int(patient_dist.get(0, 0)),
                'unstable': int(patient_dist.get(1, 0))
            }
        },
        'performance': metrics,
        'architecture': {
            'type': 'BiGRU with separated feature paths',
            'bigru_layers': [64, 32],
            'attention_units': 64,
            'dense_layers': [32],
            'dropout_rate': 0.3,
            'optimizer': 'Adam (amsgrad=True)',
            'learning_rate': 0.001,
            'static_embedding_dim': 32
        },
        'training_details': {
            'sliding_window_sizes': WINDOW_SIZES,
            'class_weight_applied': True,
            'scaler': 'StandardScaler (train-only fit)',
            'early_stopping': f'val_auc, patience={PATIENCE}'
        },
        'xai_capabilities': [
            'SHAP feature attribution (DeepExplainer)',
            'Bahdanau temporal attention weights',
            'Monte Carlo Dropout uncertainty (20 passes)',
            'Counterfactual / What-If analysis',
            'Natural language explanations',
            'Temporal feature contribution heatmaps'
        ],
        'limitations': [
            'Trained on synthetic data — real-world clinical validation required',
            'Performance may vary outside training age range (18-89)',
            'Not validated for pediatric hemodialysis patients',
            'Attention weights provide correlation, not strict causation'
        ],
        'ethical_considerations': [
            'Model should support, not replace, clinical judgment',
            'Risk predictions must always be reviewed by qualified clinicians',
            'False negatives (missed instability) carry higher clinical risk than false positives'
        ],
        'framework': f'TensorFlow {tf.__version__}',
        'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }
    
    card_path = os.path.join(output_dir, 'model_card.json')
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    print(f"  Model card saved: {card_path}")
    
    print("\n✅ All artifacts saved successfully!")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train DialysisGuard BiGRU Model v2')
    parser.add_argument('--validate', action='store_true',
                        help='Run full validation after training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable sliding window augmentation')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DialysisGuard Model Training Pipeline v2")
    print("=" * 60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU detected: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    else:
        print("\n⚠️  No GPU detected, training on CPU")
    
    # Step 1: Load data
    df = load_data(DATA_PATH)
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Encode features
    df, label_encoders = encode_features(df, fit=True)
    
    # Step 4: Prepare sequences (NO scaling yet — scale after split)
    X, y, y_per_step, feature_names = prepare_sequences(df)
    
    # Step 5: Split data (y_per_step split together to stay aligned)
    X_train, X_val, X_test, y_train, y_val, y_test, y_ps_train = split_data(
        X, y, y_per_step=y_per_step
    )
    
    # Step 6: Create sliding window augmentation
    X_aug, y_aug = None, None
    if not args.no_augment:
        X_aug, y_aug = create_sliding_windows(X_train, y_ps_train)
    
    # Step 7: Scale data (fit on TRAINING SET ONLY)
    X_train, X_val, X_test, X_aug, scaler = scale_data(
        X_train, X_val, X_test, X_aug
    )
    
    # Step 8: Combine training data with augmented data
    if X_aug is not None and y_aug is not None:
        print(f"\n  Combining original training ({X_train.shape[0]}) + augmented ({X_aug.shape[0]})")
        X_train_full = np.concatenate([X_train, X_aug], axis=0)
        y_train_full = np.concatenate([y_train, y_aug], axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(X_train_full))
        X_train_full = X_train_full[indices]
        y_train_full = y_train_full[indices]
        print(f"  Final training set: {X_train_full.shape[0]} sequences")
    else:
        X_train_full = X_train
        y_train_full = y_train
    
    # Step 9: Train model
    n_static = len(STATIC_FEATURES)
    n_temporal = len(feature_names) - n_static
    
    builder, history = train_model(
        X_train_full, y_train_full, X_val, y_val,
        n_features=X.shape[2],
        n_static=n_static,
        n_temporal=n_temporal,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 10: Evaluate (on original test set — not augmented)
    metrics = evaluate_model(builder, X_test, y_test, feature_names)
    
    # Step 11: Save artifacts
    save_artifacts(builder, scaler, label_encoders, feature_names,
                   metrics, df, OUTPUT_DIR)
    
    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"  Accuracy: {metrics['accuracy']:.4f} {'✅' if metrics['accuracy'] >= 0.80 else '⚠️'}")
        print(f"  AUC:      {metrics['auc']:.4f} {'✅' if metrics['auc'] >= 0.80 else '⚠️'}")
    
    return metrics


if __name__ == '__main__':
    main()
