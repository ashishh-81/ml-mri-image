# --- Hidden Markov Model (HMM) Example ---
# Install library (only once):
# pip install hmmlearn numpy matplotlib

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Step 1: Create sample sequential data
# Example: 0 = Cold, 1 = Hot (observations)
# Hidden states: 0 = Rainy, 1 = Sunny
observations = np.array([[0], [1], [1], [0], [1], [1], [0], [0], [1], [1]])

# Step 2: Define and train the HMM model
model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=42)

# Fit the model on observed sequence
model.fit(observations)

# Step 3: Predict the hidden states for the sequence
hidden_states = model.predict(observations)

print("Observed sequence (0=Cold, 1=Hot):")
print(observations.flatten())
print("\nPredicted hidden states (0=Rainy, 1=Sunny):")
print(hidden_states)

# Step 4: Generate new sequence prediction
X_new, Z_new = model.sample(5)
print("\nGenerated future observation sequence:")
print(X_new.flatten())
print("Generated hidden states:")
print(Z_new)

# Step 5: Visualize results
plt.figure(figsize=(8,3))
plt.plot(observations, label="Observed (0=Cold, 1=Hot)", marker='o')
plt.plot(hidden_states, label="Predicted Hidden States", marker='x')
plt.xlabel("Time Step")
plt.ylabel("State")
plt.title("Hidden Markov Model - State Prediction")
plt.legend()
plt.grid(True)
plt.show()
