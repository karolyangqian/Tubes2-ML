from typing import Any, Optional

from prometheus_client import h

from scratchlayers.core.layer import Layer
import numpy as np

class LSTMScratch(Layer):
    def __init__(self, n_neurons: int, return_sequences: bool = False):
        super().__init__()
        self.n_neurons = n_neurons
        self.return_sequences = return_sequences        
        self.kernel_weights: Optional[np.ndarray[Any, Any]] = None
        self.recurrent_weights: Optional[np.ndarray[Any, Any]] = None
        self.bias: Optional[np.ndarray[Any, Any]] = None

    def build(self, input_shape: Optional[tuple[int, ...]]):
        self.input_shape = input_shape
        
        if self.kernel_weights is None:
            raise ValueError("Kernel weights not set")
        if self.recurrent_weights is None:
            raise ValueError("Recurrent weights not set")
        if self.bias is None:
            raise ValueError("Bias not set")
        
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
        
        if self.kernel_weights is None:
            raise ValueError("Kernel weights not set.")
        if self.recurrent_weights is None:
            raise ValueError("Recurrent weights not set.")
        if self.bias is None:
            raise ValueError("Bias not set.")
        
        h_t = np.zeros((x.shape[0], self.n_neurons))
        c_t = np.zeros((x.shape[0], self.n_neurons))
        
        batch_size, ts, _ = x.shape
        
        outputs = np.zeros((batch_size, ts, self.n_neurons))
        
        for i in range(ts):
            
            x_t = x[:, i, :]
            
            z = np.dot(x_t, self.kernel_weights) + np.dot(h_t, self.recurrent_weights) + self.bias
            
            i_t = z[:, :self.n_neurons]
            f_t = z[:, self.n_neurons:2*self.n_neurons]
            c_tilde = z[:, 2*self.n_neurons:3*self.n_neurons]
            o_t = z[:, 3*self.n_neurons:4*self.n_neurons]
            
            i_t = self._activation("sigmoid").forward(i_t)
            f_t = self._activation("sigmoid").forward(f_t)
            c_tilde = self._activation("tanh").forward(c_tilde)
            o_t = self._activation("sigmoid").forward(o_t)
            
            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * self._activation("tanh").forward(c_t)

            outputs[:, i, :] = h_t

        if self.return_sequences:
            return outputs
        else:
            return h_t

    # Setter ---------------------------------------------------------------
    
    def set_kernel(self, weights: np.ndarray[Any, Any]):
        self.kernel_weights = weights
    
    def set_recurrent_kernel(self, weights: np.ndarray[Any, Any]):
        self.recurrent_weights = weights
        
    def set_bias(self, bias: np.ndarray[Any, Any]):
        self.bias = bias
        
        
    # Getter ---------------------------------------------------------------
        
    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` method first.")
        return self.output_shape
        
    def get_kernel_i(self) -> np.ndarray[Any, Any]:
        if self.kernel_weights is None:
            raise ValueError("Kernel weights not set.")
        return self.kernel_weights[:, :self.n_neurons]
    
    def get_kernel_f(self) -> np.ndarray[Any, Any]:
        if self.kernel_weights is None:
            raise ValueError("Kernel weights not set.")
        return self.kernel_weights[:, self.n_neurons:2*self.n_neurons]
    
    def get_kernel_o(self) -> np.ndarray[Any, Any]:
        if self.kernel_weights is None:
            raise ValueError("Kernel weights not set.")
        return self.kernel_weights[:, 2*self.n_neurons:3*self.n_neurons]
    
    def get_kernel_c(self) -> np.ndarray[Any, Any]:
        if self.kernel_weights is None:
            raise ValueError("Kernel weights not set.")
        return self.kernel_weights[:, 3*self.n_neurons:4*self.n_neurons]