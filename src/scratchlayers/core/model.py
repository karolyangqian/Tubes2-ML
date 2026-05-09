from scratchlayers.core.layer import Layer
import numpy as np
from typing import Any

class Model(Layer):
    def __init__(self):
        super().__init__()
    
    def predict(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return self.forward(x)