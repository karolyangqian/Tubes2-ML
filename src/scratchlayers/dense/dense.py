import numpy as np
from typing import Optional

from scratchlayers.core.activation import Activation
from scratchlayers.core.layer import Layer

class Dense(Layer):
    def __init__(self, n_neurons: int, activation: str = "relu"):
        super().__init__()
        self.n_inputs: Optional[int] = None
        self.n_neurons = n_neurons
        
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        
        self.inputs: Optional[np.ndarray] = None
        self.outputs: Optional[np.ndarray] = None
        self.nets: Optional[np.ndarray] = None
        
        self.activation_name = activation
        self.activation: Optional[Activation] = None
        
        self.output_shape: Optional[tuple[int, ...]] = None
        
    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        if input_shape is None:
            raise ValueError("Input shape cannot be None")
        
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"Dense layer requires a 2D input shape (batch_size, features), got: {input_shape}")
        
        self.input_shape = input_shape
        
        if self.weights is None:
            raise ValueError("Input weights not set")
        if self.biases is None:
            raise ValueError("Input bias not set")
        
        self.activation = self._activation(self.activation_name)
        
        batch_size, _ = input_shape
        self.output_shape = (batch_size, self.n_neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.biases is None:
            raise ValueError("Layer weights and biases must be initialized before forward pass.")
        
        if self.activation is None:
            raise ValueError("Activation function must be initialized before forward pass.")
        
        self.inputs = x
        z = np.dot(x, self.weights) + self.biases
        self.outputs = self.activation.forward(z)
        return self.outputs

    def get_output_shape(self) -> Optional[tuple[int, ...]]:
        return self.output_shape
