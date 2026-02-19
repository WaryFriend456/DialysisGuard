"""
Attention-Augmented BiGRU Model for Hemodialysis Instability Prediction.

Architecture (v2 — Per-Timestep):
    Static Features → Dense(32) → Repeat across timesteps ─┐
                                                            ├─ Concat
    Temporal Features → BatchNorm ──────────────────────────┘
                                                            ↓
    BiGRU(64, return_sequences=True) → BatchNorm → Dropout
                                                            ↓
    BiGRU(32, return_sequences=True) → BatchNorm → Dropout
                                                            ↓
    ★ Bahdanau Attention ★ (for XAI — extracts temporal weights)
                                                            ↓
    Dense(32, relu) → Dropout → Dense(1, sigmoid)

Changes from v1:
    - Per-timestep prediction (return_sequences on final output)
    - Bidirectional GRU for better temporal capture
    - Separated static/temporal feature paths
    - Removed RepeatVector(1) → GRU(32) bottleneck
    - Attention now used purely for XAI weight extraction
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import numpy as np


class BahdanauAttention(layers.Layer):
    """
    Bahdanau (Additive) Attention mechanism.
    
    Computes attention weights over time steps, indicating which moments
    in the dialysis session the model focuses on for its prediction.
    
    Returns:
        context_vector: Weighted sum of inputs (batch, features)
        attention_weights: Per-timestep importance (batch, timesteps, 1)
    """
    
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
    
    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        # score shape: (batch, timesteps, 1)
        score = self.V(tf.nn.tanh(self.W(inputs)))
        
        # attention_weights shape: (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch, features)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        
        # Always return both — context for prediction, weights for XAI
        return context_vector, attention_weights
    
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.units})
        return config


class AttentionGRUModel:
    """
    Builds and manages the Attention-Augmented BiGRU model (v2).
    
    Key improvements over v1:
    - Bidirectional GRU layers
    - Separated static/temporal feature processing
    - Per-sequence classification (context from attention → Dense → sigmoid)
    - No RepeatVector bottleneck
    """
    
    def __init__(self, n_timesteps, n_features, n_static=21, n_temporal=11,
                 gru_units=(64, 32), attention_units=64,
                 dense_units=(32,), dropout_rate=0.3, learning_rate=0.001):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_static = n_static
        self.n_temporal = n_temporal
        self.gru_units = gru_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.attention_model = None
    
    def build(self):
        """Build the full model with separated feature paths."""
        inputs = layers.Input(shape=(self.n_timesteps, self.n_features),
                              name='sequence_input')
        
        # === Separate static and temporal features ===
        # Static features: first n_static columns (constant per patient)
        static_slice = layers.Lambda(
            lambda x: x[:, 0, :self.n_static],  # Take from first timestep (they're constant)
            name='static_slice'
        )(inputs)
        
        # Temporal features: remaining columns (vary per timestep)
        temporal_slice = layers.Lambda(
            lambda x: x[:, :, self.n_static:],
            name='temporal_slice'
        )(inputs)
        
        # === Static feature embedding ===
        static_embed = layers.Dense(32, activation='relu', name='static_embed')(static_slice)
        static_embed = layers.Dropout(self.dropout_rate * 0.5, name='static_dropout')(static_embed)
        # Repeat across timesteps for concatenation
        static_repeated = layers.RepeatVector(self.n_timesteps, name='static_repeat')(static_embed)
        
        # === Temporal feature normalization ===
        temporal_normed = layers.BatchNormalization(name='temporal_bn')(temporal_slice)
        
        # === Concatenate static embedding + temporal features ===
        combined = layers.Concatenate(name='combine_features')([static_repeated, temporal_normed])
        
        # === BiGRU Layer 1 ===
        x = layers.Bidirectional(
            layers.GRU(self.gru_units[0], return_sequences=True, name='gru_1'),
            name='bigru_1'
        )(combined)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        # === BiGRU Layer 2 ===
        x = layers.Bidirectional(
            layers.GRU(self.gru_units[1], return_sequences=True, name='gru_2'),
            name='bigru_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # === Attention Layer (produces context vector + attention weights) ===
        attention_layer = BahdanauAttention(self.attention_units, name='attention')
        context, attn_weights = attention_layer(x)
        
        # === Classification Head ===
        z = layers.Dense(self.dense_units[0], activation='relu', name='dense_1')(context)
        z = layers.Dropout(self.dropout_rate, name='dropout_dense_1')(z)
        
        # Output: single probability per sequence
        outputs = layers.Dense(1, activation='sigmoid',
                               name='prediction', dtype='float32')(z)
        
        self.model = Model(inputs=inputs, outputs=outputs,
                           name='DialysisGuard_AttentionBiGRU_v2')
        
        # Build attention extraction model (shares layers with main model)
        self.attention_model = Model(
            inputs=inputs, outputs=attn_weights,
            name='attention_extractor'
        )
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=True
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return self.model
    
    def get_attention_weights(self, X):
        """
        Extract attention weights for input sequences.
        
        Args:
            X: Input sequences (batch, timesteps, features)
            
        Returns:
            Attention weights (batch, timesteps, 1) - how much each
            time step contributed to the prediction.
        """
        if self.attention_model is None:
            raise ValueError("Model must be built first")
        return self.attention_model.predict(X, verbose=0)
    
    def predict_with_dropout(self, X, n_passes=20):
        """
        Monte Carlo Dropout: Run inference with dropout active N times.
        
        Returns mean prediction, std, and confidence interval.
        This quantifies epistemic uncertainty in the prediction.
        """
        predictions = []
        for _ in range(n_passes):
            # training=True keeps dropout active during inference
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # (n_passes, batch, 1)
        
        mean = np.mean(predictions, axis=0).squeeze()
        std = np.std(predictions, axis=0).squeeze()
        
        # 95% confidence interval
        ci_lower = np.maximum(0, mean - 1.96 * std)
        ci_upper = np.minimum(1, mean + 1.96 * std)
        
        return {
            'mean': float(mean) if np.isscalar(mean) or mean.ndim == 0 else mean.tolist(),
            'std': float(std) if np.isscalar(std) or std.ndim == 0 else std.tolist(),
            'ci_lower': float(ci_lower) if np.isscalar(ci_lower) or ci_lower.ndim == 0 else ci_lower.tolist(),
            'ci_upper': float(ci_upper) if np.isscalar(ci_upper) or ci_upper.ndim == 0 else ci_upper.tolist(),
            'all_predictions': predictions.squeeze().tolist()
        }
    
    def get_callbacks(self, patience=10, checkpoint_path=None):
        """Get training callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=patience,
                mode='max',
                restore_best_weights=False,  # False to avoid deepcopy/recursion issues
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if checkpoint_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def build_model(n_timesteps, n_features, **kwargs):
    """Convenience function to build and return model."""
    builder = AttentionGRUModel(n_timesteps, n_features, **kwargs)
    model = builder.build()
    return builder
