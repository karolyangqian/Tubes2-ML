import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from typing import Optional
from scratchlayers.core.activation import Activation, ReLU, Sigmoid, Linear, Tanh, Softmax

class Layer(ABC):
    def __init__(self):
        self.input_shape: Optional[tuple[int, ...]] = None
        self.output_shape: Optional[tuple[int, ...]] = None

    @abstractmethod
    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        raise NotImplementedError("`build()` method not implemented yet")

    @abstractmethod
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raise NotImplementedError("`forward()` method not implemented yet")
    
    @abstractmethod
    def get_output_shape(self) -> Optional[tuple[int, ...]]:
        raise NotImplementedError("`get_output_shape()` method not implemented yet")
    
    def add_weight(self, shape: tuple[int, ...], initializer: Optional[str] = None) -> np.ndarray[Any, Any]:
        if initializer == 'zeros':
            return np.zeros(shape)
        elif initializer == 'ones':
            return np.ones(shape)
        elif initializer == 'random':
            return np.random.rand(*shape)
        else:
            return np.random.rand(*shape)
    
    def _activation(self, activation_name: str) -> Activation:
        if activation_name == "relu":
            return ReLU()
        elif activation_name == "sigmoid":
            return Sigmoid()
        elif activation_name == "linear":
            return Linear()
        elif activation_name == "tanh":
            return Tanh()
        elif activation_name == "softmax":
            return Softmax()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        

