import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import MRICNN

# -------- SETTINGS --------
model_path = "mri_cnn_model.pth"
class_names = ["glioma", "meningioma", "no tumor", "pituitary"]
image_path = "new_images/predict.jpg"   # <-- path to your new MRI image

# -------- LOAD TRAINED MODEL --------
model = MRICNN(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# -------- IMAGE PREPROCESSING --------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(path):
    """Predict tumor type for a given MRI image."""
    image = Image.open(path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_label = class_names[predicted.item()]
    return pred_label

# -------- RUN PREDICTION --------
predicted_label = predict_image(image_path)
print(f"🧠 Prediction for {image_path}: {predicted_label}")

# -------- SHOW IMAGE --------
image = Image.open(image_path)
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Tumor Type: {predicted_label}")
plt.axis("off")
plt.show()
