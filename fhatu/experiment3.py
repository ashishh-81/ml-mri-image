def load_images_from_folder(root_folder, img_size=(128, 128)):
    X = []
    y = []
    class_names = []
    subfolders = []

    # Search through all subfolders (Training, Testing, etc.)
    for split in os.listdir(root_folder):
        split_path = os.path.join(root_folder, split)
        if not os.path.isdir(split_path):
            continue

        # Inside each split (Training/Testing), check each class
        for class_name in os.listdir(split_path):
            folder = os.path.join(split_path, class_name)
            if not os.path.isdir(folder):
                continue
            if class_name not in class_names:
                class_names.append(class_name)
            label = class_names.index(class_name)

            # Load all image files in the class folder
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                if not (file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.jpeg')):
                    continue
                try:
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img)
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"⚠️ Skipping {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} images from {len(class_names)} classes: {class_names}")
    return X, y, class_names
