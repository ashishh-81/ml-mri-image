import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --------------------------
# Define the exact same model as training
# --------------------------
class MRICNN(nn.Module):
    def __init__(self, num_classes=4):
        super(MRICNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # <- 1 channel for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------------------------
# Load trained model
# --------------------------
@st.cache_resource
def load_model():
    model = MRICNN(num_classes=4)
    model.load_state_dict(torch.load("mri_cnn_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_model()
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --------------------------
# Function to check if uploaded image is valid MRI
# --------------------------
def is_valid_mri(image):
    """
    Simple image check:
    - Ensures it's grayscale
    - Rejects overly bright/dark/colorful images
    """
    img_array = np.array(image)

    # Reject color images with high color variance
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        color_var = np.var(img_array[:, :, 0] - img_array[:, :, 1]) + np.var(img_array[:, :, 1] - img_array[:, :, 2])
        if color_var > 500:  # too colorful to be MRI
            return False

    # For grayscale — check brightness
    gray_mean = np.mean(img_array)
    if gray_mean < 30 or gray_mean > 220:  # too dark or too bright
        return False

    return True


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="MRI Brain Tumor Classification", page_icon="🧠", layout="centered")

st.title("🧠 MRI Brain Tumor Classification")
st.markdown("Upload an MRI image to predict the tumor type using the trained CNN model.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Check if the image looks like an MRI ---
    if not is_valid_mri(img):
        st.error("⚠️ Invalid image! Please upload a valid brain MRI scan.")
    else:
        # Transform for model input
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img_t = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img_t)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()

        result = classes[pred_class]
        confidence = probs[pred_class].item() * 100

        # Display results
        st.markdown("---")
        st.subheader(f"🎯 Prediction: **{result}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.markdown("### 🔍 Class Probabilities:")
        for i, cls in enumerate(classes):
            st.progress(float(probs[i]))
            st.text(f"{cls}: {probs[i]*100:.2f}%")
