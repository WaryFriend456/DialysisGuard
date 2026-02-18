"""
DialysisGuard Model Training Pipeline

Trains an Attention-Augmented GRU model on synthetic_hemodialysis_timeseries.csv
for hemodialysis instability prediction.

Usage:
    python train_model.py                  # Train and save model
    python train_model.py --validate       # Train, validate, and print metrics

Output artifacts (saved to backend/ml/):
    - gru_model.h5           : Trained model weights
    - scaler.pkl             : Fitted StandardScaler
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
    print(f"  Target distribution: {dict(df[TARGET].value_counts())}")
    
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
    Prepare sequences for the GRU model.
    
    Each patient's 30 time-step record becomes one sequence.
    Features are scaled using StandardScaler.
    
    Returns:
        X: (n_patients, 30, n_features)
        y: (n_patients,) - 1 if patient had ANY unstable event
        scaler: fitted scaler
        feature_names: list of feature names
    """
    print("Preparing sequences...")
    
    # All features to use (static + temporal + engineered)
    engineered_features = [
        'BP_Change', 'HR_Change', 'BP_Deviation', 'HR_Deviation',
        'BP_Volatility', 'HR_Volatility', 'Fluid_Rate_Per_Kg',
        'Session_Progress'
    ]
    
    all_features = STATIC_FEATURES + TEMPORAL_FEATURES + engineered_features
    feature_names = all_features.copy()
    
    # Sort and group by patient
    df = df.sort_values(['Patient_ID', 'Time_Minutes'])
    patients = df['Patient_ID'].unique()
    
    X_list = []
    y_list = []
    
    for pid in patients:
        patient_data = df[df['Patient_ID'] == pid]
        
        # Get feature matrix for this patient (30 × n_features)
        X_patient = patient_data[all_features].values
        
        # Target: 1 if patient has ANY unstable step in their session
        # This gives us a per-patient label for sequence classification
        y_patient = int(patient_data[TARGET].max())
        
        X_list.append(X_patient)
        y_list.append(y_patient)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"  Sequence shape: {X.shape}")
    print(f"  Target distribution: stable={int((y == 0).sum())}, unstable={int((y == 1).sum())}")
    
    # Scale features
    n_patients, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)
    
    X = X_scaled.reshape(n_patients, n_timesteps, n_features).astype(np.float32)
    
    # Handle any NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Final X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, scaler, feature_names


def split_data(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=42):
    """Split data by patient (no data leakage)."""
    print("Splitting data...")
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=random_state, stratify=y_trainval
    )
    
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Training
# ============================================================

def train_model(X_train, y_train, X_val, y_val, n_features,
                epochs=None, batch_size=None):
    """Build and train the Attention-Augmented GRU model."""
    _epochs = epochs or EPOCHS
    _batch_size = batch_size or BATCH_SIZE
    
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    builder = AttentionGRUModel(
        n_timesteps=SEQUENCE_LENGTH,
        n_features=n_features,
        gru_units=(128, 64, 32),
        attention_units=64,
        dense_units=(64, 32),
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
    class_weight = {0: 1.0, 1: n_neg / n_pos if n_pos > 0 else 1.0}
    print(f"  Class weights: {class_weight}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=_epochs,
        batch_size=_batch_size,
        class_weight=class_weight,
        callbacks=builder.get_callbacks(patience=PATIENCE),
        verbose=1
    )
    
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
    print(f"    Mean predictions: {[f'{v:.3f}' for v in (mc_results['mean'] if isinstance(mc_results['mean'], list) else [mc_results['mean']])]}")
    print(f"    Std deviations:   {[f'{v:.3f}' for v in (mc_results['std'] if isinstance(mc_results['std'], list) else [mc_results['std']])]}")
    
    # Test attention weights
    print("\n  Testing attention weight extraction...")
    attn_weights = builder.get_attention_weights(sample)
    print(f"    Attention shape: {attn_weights.shape}")
    print(f"    Weights sum per sample: {[f'{w:.4f}' for w in attn_weights.squeeze().sum(axis=-1) if np.isscalar(w) or w.ndim == 0][:3]}")
    
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
    
    # 1. Save model
    model_path = os.path.join(output_dir, 'gru_model.h5')
    builder.model.save(model_path)
    print(f"  Model saved: {model_path}")
    
    # 2. Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")
    
    # 3. Save label encoders
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"  Label encoders saved: {encoders_path}")
    
    # 4. Save feature config
    feature_config = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_timesteps': SEQUENCE_LENGTH,
        'static_features': STATIC_FEATURES,
        'temporal_features': TEMPORAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'boolean_features': BOOLEAN_FEATURES,
        'engineered_features': [
            'BP_Change', 'HR_Change', 'BP_Deviation', 'HR_Deviation',
            'BP_Volatility', 'HR_Volatility', 'Fluid_Rate_Per_Kg',
            'Session_Progress'
        ],
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
    target_dist = dict(df.groupby('Patient_ID')[TARGET].max().value_counts())
    
    model_card = {
        'model_name': 'DialysisGuard Attention-GRU v1.0',
        'model_type': 'Attention-Augmented GRU (128→64→Attention→32)',
        'task': 'Binary classification — Hemodialysis instability prediction',
        'training_dataset': {
            'source': 'synthetic_hemodialysis_timeseries.csv',
            'total_rows': len(df),
            'patients': n_patients,
            'time_steps_per_patient': SEQUENCE_LENGTH,
            'features': len(feature_names),
            'target_distribution': {
                'stable': int(target_dist.get(0, 0)),
                'unstable': int(target_dist.get(1, 0))
            }
        },
        'performance': metrics,
        'architecture': {
            'gru_layers': [128, 64, 32],
            'attention_units': 64,
            'dense_layers': [64, 32],
            'dropout_rate': 0.3,
            'optimizer': 'Adam (amsgrad=True)',
            'learning_rate': 0.001
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
    parser = argparse.ArgumentParser(description='Train DialysisGuard GRU Model')
    parser.add_argument('--validate', action='store_true',
                        help='Run full validation after training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DialysisGuard Model Training Pipeline")
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
    
    # Step 4: Prepare sequences
    X, y, scaler, feature_names = prepare_sequences(df, fit_scaler=True)
    
    # Step 5: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Step 6: Train model
    builder, history = train_model(X_train, y_train, X_val, y_val,
                                    n_features=X.shape[2],
                                    epochs=args.epochs,
                                    batch_size=args.batch_size)
    
    # Step 7: Evaluate
    metrics = evaluate_model(builder, X_test, y_test, feature_names)
    
    # Step 8: Save artifacts
    save_artifacts(builder, scaler, label_encoders, feature_names,
                   metrics, df, OUTPUT_DIR)
    
    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"  Accuracy: {metrics['accuracy']:.4f} {'✅' if metrics['accuracy'] >= 0.85 else '⚠️'}")
        print(f"  AUC:      {metrics['auc']:.4f} {'✅' if metrics['auc'] >= 0.80 else '⚠️'}")
    
    return metrics


if __name__ == '__main__':
    main()
