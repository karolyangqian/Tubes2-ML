from typing import Any, Optional

from scratchlayers.core.layer import Layer
from scratchlayers.core.activation import Activation, ReLU, Sigmoid, Linear, Tanh, Softmax
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
        
        self.activation_name = activation
        self.activation: Optional[Activation] = None
        
    def build(self, input_shape: Optional[tuple[int, ...]]):
        self. input_shape = input_shape
        
        if self.input_weights is None:
            raise ValueError("Input weights not set")
        if self.input_bias is None:
            raise ValueError("Input bias not set")
        if self.recurrent_weights is None:
            raise ValueError("Recurrent weights not set")
        
        self.activation = self._activation()
        
        if input_shape is not None:
            if self.return_sequences and len(input_shape) == 3:
                batch_size, timesteps, _ = input_shape
                self.output_shape = (batch_size, timesteps, self.n_neurons)
            elif not self.return_sequences and len(input_shape) == 3:
                batch_size, _, _ = input_shape
                self.output_shape = (batch_size, self.n_neurons)
            elif not self.return_sequences and len(input_shape) == 2:
                batch_size, _ = input_shape
                self.output_shape = (batch_size, self.n_neurons)
            else:
                raise ValueError("Invalid input shape for LSTM layer")  
    
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raise NotImplementedError("Forward pass not implemented yet")
    
    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` method first.")
        return self.output_shape 
    
    def _activation(self) -> Activation:
        if self.activation_name == "relu":
            return ReLU()
        elif self.activation_name == "sigmoid":
            return Sigmoid()
        elif self.activation_name == "linear":
            return Linear()
        elif self.activation_name == "tanh":
            return Tanh()
        elif self.activation_name == "softmax":
            return Softmax()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        