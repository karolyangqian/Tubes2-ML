import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Calculates the output of the activation function for the input x"""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the activation function with respect to x"""
        pass
    
class Linear(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)

class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x).reshape(-1, 1)
        # Jacobian matrix
        return np.diagflat(s) - np.dot(s, s.T)

class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        t = self.forward(x)
        return 1 - np.square(t)