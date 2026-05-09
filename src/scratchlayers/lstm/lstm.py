from typing import Any, Optional

from scratchlayers.core.layer import Layer
import numpy as np

class LSTMScratch(Layer):
    def __init__(self, n_neurons: int, return_sequences: bool = False):
        super().__init__()
        self.n_neurons = n_neurons
        self.return_sequences = return_sequences        
        self.forget_gate_weights: Optional[np.ndarray[Any, Any]] = None
        self.input_gate_weights: Optional[np.ndarray[Any, Any]] = None
        self.candidate_weights: Optional[np.ndarray[Any, Any]] = None
        self.output_gate_weights: Optional[np.ndarray[Any, Any]] = None
        self.forget_gate_bias: Optional[np.ndarray[Any, Any]] = None
        self.input_gate_bias: Optional[np.ndarray[Any, Any]] = None
        self.candidate_bias: Optional[np.ndarray[Any, Any]] = None
        self.output_gate_bias: Optional[np.ndarray[Any, Any]] = None

    def build(self, input_shape: Optional[tuple[int, ...]]):
        self.input_shape = input_shape
        
        if self.forget_gate_weights is None:
            raise ValueError("Forget gate weights not set")
        if self.input_gate_weights is None:
            raise ValueError("Input gate weights not set")
        if self.candidate_weights is None:
            raise ValueError("Candidate weights not set")
        if self.output_gate_weights is None:
            raise ValueError("Output gate weights not set")
        if self.forget_gate_bias is None:
            raise ValueError("Forget gate bias not set")
        if self.input_gate_bias is None:
            raise ValueError("Input gate bias not set")
        if self.candidate_bias is None:
            raise ValueError("Candidate bias not set")
        if self.output_gate_bias is None:
            raise ValueError("Output gate bias not set")
        
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
        # TODO: Implement the forward pass of the LSTM cell
        raise NotImplementedError("Forward pass not implemented yet")
    
    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` method first.")
        return self.output_shape
    
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