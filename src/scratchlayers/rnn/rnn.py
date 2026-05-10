from typing import Any, Optional

from scratchlayers.core.layer import Layer
from scratchlayers.core.activation import Activation
import numpy as np

class SimpleRNNScratch(Layer):
    def __init__(self, n_neurons: int, activation: str = "relu", return_sequences: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.return_sequences = return_sequences   
        self.input_weights: Optional[np.ndarray[Any, Any]] = None
        self.input_bias: Optional[np.ndarray[Any, Any]] = None
        self.recurrent_weights: Optional[np.ndarray[Any, Any]] = None
        self.output_shape: Optional[tuple[int, ...]] = None
        
        self.inputs: Optional[np.ndarray] = None
        self.outputs: Optional[np.ndarray] = None
        
        self.activation_name = activation
        self.activation: Optional[Activation] = None
        
    def build(self, input_shape: Optional[tuple[int, ...]]):
        self.input_shape = input_shape
        
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"SimpleRNN requires a 3D input shape (batch_size, timesteps, features), got: {input_shape}")
        if self.input_weights is None:
            raise ValueError("Input weights not set")
        if self.input_bias is None:
            raise ValueError("Input bias not set")
        if self.recurrent_weights is None:
            raise ValueError("Recurrent weights not set")
        
        self.activation = self._activation(self.activation_name)
        
        if self.return_sequences:
            batch_size, timesteps, _ = input_shape
            self.output_shape = (batch_size, timesteps, self.n_neurons)
        else:
            batch_size, _, _ = input_shape
            self.output_shape = (batch_size, self.n_neurons)
    
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.activation is None:
            raise ValueError("Activation function not set. Call `build()` method first.")
        if self.input_weights is None:
            raise ValueError("Input weights not set.")
        if self.input_bias is None:
            raise ValueError("Input bias not set.")
        if self.recurrent_weights is None:
            raise ValueError("Recurrent weights not set.")
        
        self.inputs = x
        
        batch_size, timesteps, _ = x.shape
        
        h_t = np.zeros((batch_size, self.n_neurons))
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, timesteps, self.n_neurons))
            
        for t in range(timesteps):
            x_t = x[:, t, :]
            h_t = self.activation.forward(
                np.dot(x_t, self.input_weights) + 
                np.dot(h_t, self.recurrent_weights) + 
                self.input_bias
            )
            if self.return_sequences:
                outputs[:, t, :] = h_t
        
        self.outputs = outputs if self.return_sequences else h_t
        if self.return_sequences:
            return outputs
        
        return h_t
    
    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` method first.")
        return self.output_shape 
    