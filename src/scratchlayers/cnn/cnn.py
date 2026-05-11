import numpy as np
from typing import Any, Optional, Tuple, Union
from scratchlayers.core.layer import Layer
from scratchlayers.core.activation import Activation

class Conv2DScratch(Layer):
    """
    2D convolution dengan shared parameters.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = 'valid',
        activation: str = 'linear',
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size: Tuple[int, int] = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
        )
        self.strides: Tuple[int, int] = (
            (strides, strides) if isinstance(strides, int) else (strides[0], strides[1])
        )
        self.padding = padding.lower()
        self.activation_name = activation
        self.activation: Optional[Activation] = None
        self.kernel: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self._out_H: Optional[int] = None
        self._out_W: Optional[int] = None
        self._pad_h: int = 0
        self._pad_w: int = 0

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if self.kernel is None:
            raise ValueError("Kernel weights not set")
        if self.bias is None:
            raise ValueError("Bias not set")
        self.activation = self._activation(self.activation_name)
        if input_shape is not None:
            N, H, W, _ = input_shape
            kH, kW = self.kernel_size
            sh, sw = self.strides
            if self.padding == 'same':
                self._out_H = int(np.ceil(H / sh))
                self._out_W = int(np.ceil(W / sw))
                self._pad_h = max((self._out_H - 1) * sh + kH - H, 0)
                self._pad_w = max((self._out_W - 1) * sw + kW - W, 0)
            else:
                self._out_H = (H - kH) // sh + 1
                self._out_W = (W - kW) // sw + 1
            self.output_shape = (N, self._out_H, self._out_W, self.filters)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.bias is None:
            raise ValueError("Weights not set. Call `build()` first.")
        if self._out_H is None or self._out_W is None:
            raise ValueError("Output shape not computed. Call `build()` first.")
        if self.activation is None:
            raise ValueError("Activation not set. Call `build()` first.")
        N = x.shape[0]
        sh, sw = self.strides
        kH, kW = self.kernel_size

        if self.padding == 'same':
            x = np.pad(
                x,
                ((0, 0), (self._pad_h // 2, self._pad_h - self._pad_h // 2),
                 (self._pad_w // 2, self._pad_w - self._pad_w // 2), (0, 0)),
            )

        output = np.zeros((N, self._out_H, self._out_W, self.filters))
        for i in range(self._out_H):
            for j in range(self._out_W):
                patch = x[:, i * sh:i * sh + kH, j * sw:j * sw + kW, :]
                output[:, i, j, :] = np.tensordot(patch, self.kernel, axes=([1, 2, 3], [0, 1, 2])) + self.bias

        return self.activation.forward(output)

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape

class LocallyConnected2DScratch(Layer):
    """
    2D locally connected layer tanpa shared parameter.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        activation: str = 'linear',
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size: Tuple[int, int] = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
        )
        self.strides: Tuple[int, int] = (
            (strides, strides) if isinstance(strides, int) else (strides[0], strides[1])
        )
        self.activation_name = activation
        self.activation: Optional[Activation] = None
        self.kernel: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self._out_H: Optional[int] = None
        self._out_W: Optional[int] = None

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if self.kernel is None:
            raise ValueError("Kernel weights not set")
        if self.bias is None:
            raise ValueError("Bias not set")
        self.activation = self._activation(self.activation_name)
        if input_shape is not None:
            N, H, W, _ = input_shape
            kH, kW = self.kernel_size
            sh, sw = self.strides
            self._out_H = (H - kH) // sh + 1
            self._out_W = (W - kW) // sw + 1
            self.output_shape = (N, self._out_H, self._out_W, self.filters)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.bias is None:
            raise ValueError("Weights not set. Call `build()` first.")
        if self._out_H is None or self._out_W is None:
            raise ValueError("Output shape not computed. Call `build()` first.")
        if self.activation is None:
            raise ValueError("Activation not set. Call `build()` first.")
        N = x.shape[0]
        kH, kW = self.kernel_size
        sh, sw = self.strides

        output = np.zeros((N, self._out_H, self._out_W, self.filters))
        for i in range(self._out_H):
            for j in range(self._out_W):
                idx = i * self._out_W + j
                patch = x[:, i * sh:i * sh + kH, j * sw:j * sw + kW, :]
                patch_flat = patch.reshape(N, -1)
                output[:, i, j, :] = np.matmul(patch_flat, self.kernel[idx]) + self.bias[idx]

        return self.activation.forward(output)

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape

class MaxPooling2DScratch(Layer):
    """Max pooling pada spatial window 2D."""

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = (2, 2),
        strides: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.pool_size: Tuple[int, int] = (
            (pool_size, pool_size) if isinstance(pool_size, int) else (pool_size[0], pool_size[1])
        )
        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides: Tuple[int, int] = (strides[0], strides[1])
        self._out_H: Optional[int] = None
        self._out_W: Optional[int] = None
        self._C: Optional[int] = None

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if input_shape is not None:
            N, H, W, C = input_shape
            ph, pw = self.pool_size
            sh, sw = self.strides
            self._out_H = (H - ph) // sh + 1
            self._out_W = (W - pw) // sw + 1
            self._C = C
            self.output_shape = (N, self._out_H, self._out_W, C)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._out_H is None or self._out_W is None or self._C is None:
            raise ValueError("Output shape not computed. Call `build()` first.")
        N = x.shape[0]
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        output = np.zeros((N, self._out_H, self._out_W, self._C))
        for i in range(self._out_H):
            for j in range(self._out_W):
                patch = x[:, i * sh:i * sh + ph, j * sw:j * sw + pw, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
                
        return output

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape


class AveragePooling2DScratch(Layer):
    """Average pooling pada spatial window 2D."""

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = (2, 2),
        strides: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.pool_size: Tuple[int, int] = (
            (pool_size, pool_size) if isinstance(pool_size, int) else (pool_size[0], pool_size[1])
        )
        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides: Tuple[int, int] = (strides[0], strides[1])
        self._out_H: Optional[int] = None
        self._out_W: Optional[int] = None
        self._C: Optional[int] = None

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if input_shape is not None:
            N, H, W, C = input_shape
            ph, pw = self.pool_size
            sh, sw = self.strides
            self._out_H = (H - ph) // sh + 1
            self._out_W = (W - pw) // sw + 1
            self._C = C
            self.output_shape = (N, self._out_H, self._out_W, C)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._out_H is None or self._out_W is None or self._C is None:
            raise ValueError("Output shape not computed. Call `build()` first.")
        N = x.shape[0]
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        output = np.zeros((N, self._out_H, self._out_W, self._C))
        for i in range(self._out_H):
            for j in range(self._out_W):
                patch = x[:, i * sh:i * sh + ph, j * sw:j * sw + pw, :]
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))
                
        return output

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape

class GlobalAveragePooling2DScratch(Layer):
    """Reduksi dimensi spasial dengan mengambil rata-rata per channel."""

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if input_shape is not None:
            N, _, _, C = input_shape
            self.output_shape = (N, C)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=(1, 2))

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape

class GlobalMaxPooling2DScratch(Layer):
    """Reduksi dimensi spasial dengan mengambil nilai maksimum per channel."""

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if input_shape is not None:
            N, _, _, C = input_shape
            self.output_shape = (N, C)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, axis=(1, 2))

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape


class FlattenScratch(Layer):
    """Flatten semua dimensi kecuali batch dimension (row-major / C order)."""

    def build(self, input_shape: Optional[tuple[int, ...]]) -> None:
        self.input_shape = input_shape
        if input_shape is not None:
            N = input_shape[0]
            flat = int(np.prod(input_shape[1:]))
            self.output_shape = (N, flat)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)

    def get_output_shape(self) -> tuple[int, ...]:
        if self.output_shape is None:
            raise ValueError("Output shape not set. Call `build()` first.")
        return self.output_shape