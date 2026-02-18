"""
Attention-Augmented GRU Model for Hemodialysis Instability Prediction.

Architecture:
    Input → GRU(128) → BatchNorm → GRU(64) → BatchNorm →
    ★ Bahdanau Attention ★ → GRU(32) → BatchNorm →
    Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid)

The Attention layer produces temporal weights showing which time steps
contributed most to the prediction - key for explainability.
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
    
    def call(self, inputs, return_attention=False):
        # inputs shape: (batch, timesteps, features)
        # score shape: (batch, timesteps, 1)
        score = self.V(tf.nn.tanh(self.W(inputs)))
        
        # attention_weights shape: (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch, features)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        
        if return_attention:
            return context_vector, attention_weights
        return context_vector
    
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.units})
        return config


class AttentionGRUModel:
    """
    Builds and manages the Attention-Augmented GRU model.
    
    The model takes sequences of shape (timesteps, features) and predicts
    instability probability. The attention layer provides temporal
    explainability by showing which time steps matter most.
    """
    
    def __init__(self, n_timesteps, n_features, gru_units=(128, 64, 32),
                 attention_units=64, dense_units=(64, 32),
                 dropout_rate=0.3, learning_rate=0.001):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.gru_units = gru_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.attention_model = None  # For extracting attention weights
    
    def build(self):
        """Build the full model with attention layer."""
        inputs = layers.Input(shape=(self.n_timesteps, self.n_features),
                              name='sequence_input')
        
        # GRU Layer 1
        x = layers.GRU(self.gru_units[0], return_sequences=True,
                        name='gru_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        # GRU Layer 2
        x = layers.GRU(self.gru_units[1], return_sequences=True,
                        name='gru_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Attention Layer - produces temporal weights
        attention_layer = BahdanauAttention(self.attention_units,
                                            name='attention')
        context = attention_layer(x)
        
        # Reshape context for GRU Layer 3 (needs 3D input)
        context_expanded = layers.RepeatVector(1, name='expand_context')(context)
        
        # GRU Layer 3
        x = layers.GRU(self.gru_units[2], return_sequences=False,
                        name='gru_3')(context_expanded)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(self.dropout_rate * 0.67, name='dropout_3')(x)
        
        # Dense layers
        x = layers.Dense(self.dense_units[0], activation='relu',
                          name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense_1')(x)
        
        x = layers.Dense(self.dense_units[1], activation='relu',
                          name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate * 0.67, name='dropout_dense_2')(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid',
                               name='prediction', dtype='float32')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs,
                           name='DialysisGuard_AttentionGRU')
        
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
        
        # Build attention extraction model
        self._build_attention_model()
        
        return self.model
    
    def _build_attention_model(self):
        """Build a secondary model that outputs attention weights."""
        if self.model is None:
            raise ValueError("Must build main model first")
        
        attention_layer = self.model.get_layer('attention')
        
        # Get the input to the attention layer (output of gru_2 + bn_2 + dropout_2)
        gru2_dropout_output = self.model.get_layer('dropout_2').output
        
        # Create a function to extract attention weights
        # We'll rebuild the attention computation to get weights
        score = attention_layer.V(tf.nn.tanh(attention_layer.W(gru2_dropout_output)))
        attention_weights = tf.nn.softmax(score, axis=1)
        
        self.attention_model = Model(
            inputs=self.model.input,
            outputs=attention_weights,
            name='attention_extractor'
        )
    
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
            self._build_attention_model()
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
    
    def get_callbacks(self, patience=10):
        """Get training callbacks."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=patience,
                mode='max',
                restore_best_weights=True,
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
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def build_model(n_timesteps, n_features, **kwargs):
    """Convenience function to build and return model."""
    builder = AttentionGRUModel(n_timesteps, n_features, **kwargs)
    model = builder.build()
    return builder
