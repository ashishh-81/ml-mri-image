import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to training and testing folders
train_path = "archive/Training"
test_path = "archive/Testing"

# Get list of classes in training set
classes = os.listdir(train_path)
print("Classes found:", classes)

# Number of images to show per class
images_per_class = 3

# Set up the plot
plt.figure(figsize=(12, 8))
img_count = 1

# Loop through each class
for class_name in classes:
    class_dir = os.path.join(train_path, class_name)
    images = os.listdir(class_dir)[:images_per_class]  # Pick first few images
    
    for img_file in images:
        img_path = os.path.join(class_dir, img_file)
        img = mpimg.imread(img_path)
        
        plt.subplot(len(classes), images_per_class, img_count)
        plt.imshow(img, cmap='gray')
        plt.title(class_name)
        plt.axis('off')
        img_count += 1

plt.tight_layout()
plt.show()







# Experiment 2: Display Summary & Statistics of MRI Dataset
#
import os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd

# Paths to dataset
train_path = "archive/Training"
test_path = "archive/Testing"

# Function to count images in each class
def dataset_summary(path):
    summary = []
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            num_images = len(os.listdir(class_dir))
            summary.append({"Class": class_name, "Images": num_images})
    return pd.DataFrame(summary)

# Training dataset stats
train_summary = dataset_summary(train_path)
print("Training Dataset Summary:")
print(train_summary)

# Testing dataset stats
test_summary = dataset_summary(test_path)
print("\nTesting Dataset Summary:")
print(test_summary)

# Pick a sample image to get dimensions and pixel stats
sample_class = os.listdir(train_path)[0]
sample_image_path = os.path.join(train_path, sample_class, os.listdir(os.path.join(train_path, sample_class))[0])
sample_img = mpimg.imread(sample_image_path)

print("\nSample Image Info:")
print(f"Shape: {sample_img.shape}")
print(f"Min pixel value: {np.min(sample_img)}")
print(f"Max pixel value: {np.max(sample_img)}")

# Overall dataset size
total_train = train_summary["Images"].sum()
total_test = test_summary["Images"].sum()
print(f"\nTotal Training Images: {total_train}")
print(f"Total Testing Images: {total_test}")
print(f"Total Dataset Size: {total_train + total_test}")
