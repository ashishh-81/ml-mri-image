# experiment9_ensemble.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------------------
# 1️⃣ Example dataset
# Replace this with your MRI features dataset
# -----------------------------------------
data = {
    'Texture': [10, 22, 13, 45, 36, 52, 23, 43, 25, 12],
    'Intensity': [100, 120, 95, 200, 180, 210, 130, 170, 110, 90],
    'Shape': [1, 2, 1, 3, 2, 3, 1, 2, 1, 1],
    'Class': ['No Tumor', 'Meningioma', 'No Tumor', 'Pituitary', 'Glioma',
              'Pituitary', 'Meningioma', 'Glioma', 'No Tumor', 'No Tumor']
}
df = pd.DataFrame(data)

X = df[['Texture', 'Intensity', 'Shape']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------------------
# 2️⃣ Random Forest Classifier
# -----------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("🔹 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# -----------------------------------------
# 3️⃣ Gradient Boosting Classifier
# -----------------------------------------
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("\n🔹 Gradient Boosting Results")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))

# -----------------------------------------
# 4️⃣ Feature Importance Visualization (Random Forest)
# -----------------------------------------
plt.figure(figsize=(6,4))
importances = rf_model.feature_importances_
plt.bar(X.columns, importances, color='skyblue')
plt.title("Feature Importance (Random Forest)")
plt.show()
