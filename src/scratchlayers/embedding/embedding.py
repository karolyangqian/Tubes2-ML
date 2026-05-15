from scratchlayers.core.layer import Layer
import numpy as np
from typing import Optional, Any

class Embedding(Layer):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings: Optional[np.ndarray[Any, Any]] = None
        self.output_shape: Optional[tuple[int, ...]] = None
        
    def build(self, input_shape: Optional[tuple[int, ...]]):
        self.input_shape = input_shape
        
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"Embedding layer requires a 2D input shape (batch_size, sequence_length), got: {input_shape}")
        
        if self.embeddings is None:
            self.embeddings = self.add_weight((self.input_dim, self.output_dim), initializer='random')
        
        batch_size, sequence_length = input_shape
        self.output_shape = (batch_size, sequence_length, self.output_dim)
    
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.embeddings is None:
            raise ValueError("Embeddings not set. Call `build()` method first.")
        
        self.inputs = x
        
        if np.any((x < 0) | (x >= self.input_dim)):
            raise ValueError(f"Word index out of bounds for input dimension {self.input_dim}")
            
        output = self.embeddings[x]
        
        return output
        
    def get_output_shape(self) -> Optional[tuple[int, ...]]:
        return self.output_shape