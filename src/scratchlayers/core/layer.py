import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from typing import Optional

class Layer(ABC):
    def __init__(self):
        self.input_shape: Optional[tuple[int, ...]] = None
        self.output_shape: Optional[tuple[int, ...]] = None

    @abstractmethod
    def build(self, input_shape: Optional[tuple[int, ...]]):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        pass
    
    def add_weight(self, shape: tuple[int, ...], initializer: Optional[str] = None) -> np.ndarray[Any, Any]:
        if initializer == 'zeros':
            return np.zeros(shape)
        elif initializer == 'ones':
            return np.ones(shape)
        elif initializer == 'random':
            return np.random.rand(*shape)
        else:
            return np.random.rand(*shape)
        
    def get_output_shape(self) -> Optional[tuple[int, ...]]:
        return self.output_shape

