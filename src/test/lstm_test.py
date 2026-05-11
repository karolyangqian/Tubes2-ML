import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

X_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

X_test = np.random.rand(10, 10, 1)
y_test = np.random.rand(10, 1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

model = Sequential([
    LSTM(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

print("Training the model...")
history = model.fit(X_train, y_train, epochs=10, verbose=1)

predictions = model.predict(X_test)

print("\nPredictions on dummy test data:")
print(predictions)

import matplotlib.pyplot as plt
import seaborn as sns

all_weights = np.concatenate([w.flatten() for w in model.get_weights()])

plt.figure(figsize=(10, 5))
sns.histplot(all_weights, kde=True, color='skyblue')
plt.title('Distribution of All Model Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.show()