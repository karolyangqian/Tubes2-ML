from typing import Any, Optional

from scratchlayers.core.layer import Layer
import numpy as np

class LSTMScratch(Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, input_shape: Optional[tuple[int, ...]]):
        self.input_shape = input_shape
        
        # TODO: Initialize weights and biases for forget gate, input gate, candidate, and output gate
        # TODO: calculate output shape based on input shape and hidden state size
    
    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # TODO: Implement the forward pass of the LSTM cell
        raise NotImplementedError("Forward pass not implemented yet")
    
    def set_forget_gate_weights(self, x: np.ndarray[Any, Any]):
        self.forget_gate_weights = x
    
    def set_input_gate_weights(self, x: np.ndarray[Any, Any]):
        self.input_gate_weights = x
    
    def set_candidate_weights(self, x: np.ndarray[Any, Any]):
        self.candidate_weights = x
    
    def set_output_gate_weights(self, x: np.ndarray[Any, Any]):
        self.output_gate_weights = x
    
    def set_forget_gate_bias(self, x: np.ndarray[Any, Any]):
        self.forget_gate_bias = x
    
    def set_input_gate_bias(self, x: np.ndarray[Any, Any]):
        self.input_gate_bias = x
    
    def set_candidate_bias(self, x: np.ndarray[Any, Any]):
        self.candidate_bias = x
    
    def set_output_gate_bias(self, x: np.ndarray[Any, Any]):
        self.output_gate_bias = x