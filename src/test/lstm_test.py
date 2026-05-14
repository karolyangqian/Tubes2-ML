import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratchlayers.core.sequential import Sequential
from scratchlayers.embedding.embedding import Embedding
from scratchlayers.lstm.lstm import LSTMScratch
from scratchlayers.dense.dense import Dense

vocab_size = 100
embedding_dim = 64
seq_length = 10
batch_size = 32

X_test = np.random.randint(0, vocab_size, size=(batch_size, seq_length))

embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
lstm_layer = LSTMScratch(n_neurons=50, return_sequences=False)
dense_layer = Dense(n_neurons=1, activation="sigmoid")

lstm_layer.set_kernel(np.random.randn(embedding_dim, 50 * 4))
lstm_layer.set_recurrent_kernel(np.random.randn(50, 50 * 4))
lstm_layer.set_bias(np.zeros(50 * 4))
dense_layer.weights = np.random.randn(50, 1)
dense_layer.biases = np.zeros(1)

model = Sequential([
    embedding_layer,
    lstm_layer,
    dense_layer
])

model.build(input_shape=(batch_size, seq_length))
predictions = model.predict(X_test)

print(f"X_test shape  : {X_test.shape}")
print(f"Pred. shape   : {predictions.shape}")
print(f"Embbeding weights shape: {embedding_layer.embeddings.shape}")
print(f"Sample preds  :\n{predictions[:5]}")