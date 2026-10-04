# experiment8_cart.py
# CART (Classification and Regression Tree) Implementation for Categorization
# Example: MRI Brain Tumor Classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# -----------------------------------------
# 1️⃣ Load Dataset (Example dataset for demo)
# Replace with your MRI features dataset if available
# -----------------------------------------
data = {
    'Texture': [10, 22, 13, 45, 36, 52, 23, 43, 25, 12],
    'Intensity': [100, 120, 95, 200, 180, 210, 130, 170, 110, 90],
    'Shape': [1, 2, 1, 3, 2, 3, 1, 2, 1, 1],
    'Class': ['No Tumor', 'Meningioma', 'No Tumor', 'Pituitary', 'Glioma',
              'Pituitary', 'Meningioma', 'Glioma', 'No Tumor', 'No Tumor']
}

df = pd.DataFrame(data)
print("📊 Dataset:\n", df, "\n")

# -----------------------------------------
# 2️⃣ Split into Features and Labels
# -----------------------------------------
X = df[['Texture', 'Intensity', 'Shape']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------------------
# 3️⃣ Create CART Model (Decision Tree)
# -----------------------------------------
model = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=4)
model.fit(X_train, y_train)

# -----------------------------------------
# 4️⃣ Predict and Evaluate
# -----------------------------------------
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------------------
# 5️⃣ Visualize the Decision Tree
# -----------------------------------------
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
plt.title("CART Decision Tree for MRI Tumor Classification")
plt.show()
