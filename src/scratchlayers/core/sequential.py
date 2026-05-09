from scratchlayers.core.layer import Layer
from scratchlayers.core.model import Model
import numpy as np
from typing import Any, Optional

class Sequential(Model):
    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = layers

    def build(self, input_shape: Optional[tuple[int, ...]]):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.output_shape
            
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    