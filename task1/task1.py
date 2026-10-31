#Task 1
#Task requirement: Dataset, Preprocessing, and Splitting
#Task overview: Load and preprocess CIFAR-10 images, convert to grayscale inputs and RGB targets normalized to [0, 1], visualize sample pairs and pixel distributions, split into 70% train, 20% validation, and 10% test sets, and batch the training data into batches of 100.

import os
import glob
import pickle
import tarfile
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Configuration parameters
TARGET_SIZE = (16, 16)
TOTAL_SAMPLES = 2000
BATCH_SIZE = 100

# 1. Locate or extract the CIFAR-10 dataset
meta_matches = glob.glob("/kaggle/input/**/batches.meta", recursive=True) + glob.glob("./**/batches.meta", recursive=True)

cifar_dir = None
if meta_matches:
    cifar_dir = os.path.dirname(meta_matches[0])
else:
    # Check all files in /kaggle/input or local workspace to find and extract the archive
    search_dirs = ["/kaggle/input", "../input", "."]
    candidate_files = []
    for base in search_dirs:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if ".git" in root or "__pycache__" in root:
                    continue
                for f in files:
                    candidate_files.append(os.path.join(root, f))

    os.makedirs("cifar_data", exist_ok=True)
    for path in candidate_files:
        try:
            with tarfile.open(path) as tar:
                names = tar.getnames()
                if any("batches.meta" in n or "data_batch" in n for n in names):
                    tar.extractall("cifar_data")
                    extracted = glob.glob("cifar_data/**/batches.meta", recursive=True)
                    if extracted:
                        cifar_dir = os.path.dirname(extracted[0])
                        break
        except Exception:
            continue

if cifar_dir is None:
    raise FileNotFoundError("Could not find CIFAR-10 dataset in /kaggle/input or local folder.")

print("Using dataset path:", cifar_dir)

# Load class names from metadata
with open(os.path.join(cifar_dir, "batches.meta"), "rb") as f:
    meta = pickle.load(f, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

# Select animal classes: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)
selected_classes = [2, 3, 4, 5, 6, 7]

raw_data = []
raw_labels = []

# Read the 5 training batches and filter for the selected classes
for i in range(1, 6):
    batch_file = os.path.join(cifar_dir, f"data_batch_{i}")
    with open(batch_file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"]
        labels = np.array(batch[b"labels"])
        
        mask = np.isin(labels, selected_classes)
        raw_data.append(data[mask])
        raw_labels.append(labels[mask])

all_data = np.concatenate(raw_data, axis=0)
all_labels = np.concatenate(raw_labels, axis=0)

# Reshape from (N, 3072) to (N, 32, 32, 3)
all_images = all_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Shuffle and pick 2000 samples
perm = np.random.permutation(len(all_images))[:TOTAL_SAMPLES]
sampled_images = all_images[perm]
sampled_labels = all_labels[perm]

# Downsample from 32x32 to 16x16 using 2x2 average pooling to keep training fast
images_16x16 = sampled_images.reshape(-1, 16, 2, 16, 2, 3).mean(axis=(2, 4))

# 2 & 3. Normalize pixel values to [0, 1] and generate grayscale inputs and RGB targets
color_targets = images_16x16.astype(np.float32) / 255.0

# Convert RGB to grayscale using standard luminance weights: Y = 0.299*R + 0.587*G + 0.114*B
grayscale_inputs = (
    0.299 * color_targets[:, :, :, 0] +
    0.587 * color_targets[:, :, :, 1] +
    0.114 * color_targets[:, :, :, 2]
)

# Flatten images into 1D feature vectors for the neural network
# Input shape: (2000, 256), Target shape: (2000, 768)
X_flat = grayscale_inputs.reshape(TOTAL_SAMPLES, -1)
Y_flat = color_targets.reshape(TOTAL_SAMPLES, -1)

print("Grayscale input shape:", X_flat.shape, "| min:", X_flat.min(), "max:", X_flat.max())
print("Color target shape:", Y_flat.shape, "| min:", Y_flat.min(), "max:", Y_flat.max())

# 4. Visualize sample grayscale/color pairs and pixel-value distributions
num_samples_to_show = 6
fig, axes = plt.subplots(2, num_samples_to_show, figsize=(12, 4))

for i in range(num_samples_to_show):
    label_name = class_names[sampled_labels[i]]
    
    # Top row: grayscale input
    axes[0, i].imshow(grayscale_inputs[i], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"Gray ({label_name})")
    axes[0, i].axis("off")
    
    # Bottom row: true color image
    axes[1, i].imshow(color_targets[i], vmin=0, vmax=1)
    axes[1, i].set_title(f"Color ({label_name})")
    axes[1, i].axis("off")

plt.suptitle("Sample Grayscale Inputs and Color Targets (16x16)")
plt.tight_layout()
plt.show()

# Plot histograms of pixel values to sanity-check normalization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(grayscale_inputs.flatten(), bins=40, color="gray", edgecolor="black", alpha=0.8)
axes[0].set_title("Grayscale Pixel Intensity Distribution")
axes[0].set_xlabel("Pixel Value [0, 1]")
axes[0].set_ylabel("Count")

axes[1].hist(color_targets[:, :, :, 0].flatten(), bins=40, color="red", alpha=0.5, label="Red")
axes[1].hist(color_targets[:, :, :, 1].flatten(), bins=40, color="green", alpha=0.5, label="Green")
axes[1].hist(color_targets[:, :, :, 2].flatten(), bins=40, color="blue", alpha=0.5, label="Blue")
axes[1].set_title("RGB Channels Pixel Intensity Distribution")
axes[1].set_xlabel("Pixel Value [0, 1]")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.show()

# 5. Split data into Training (70%), Validation (20%), and Testing (10%)
n_train = int(0.70 * TOTAL_SAMPLES)  # 1400
n_val = int(0.20 * TOTAL_SAMPLES)    # 400

X_train, Y_train = X_flat[:n_train], Y_flat[:n_train]
X_val, Y_val = X_flat[n_train:n_train + n_val], Y_flat[n_train:n_train + n_val]
X_test, Y_test = X_flat[n_train + n_val:], Y_flat[n_train + n_val:]

# Keep unflattened copies of test images for visual evaluation in Task 4
gray_test = grayscale_inputs[n_train + n_val:]
color_test = color_targets[n_train + n_val:]

print(f"Split sizes -> Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# 6. Divide training data into batches of 100 samples each
num_batches = len(X_train) // BATCH_SIZE

X_train_batches = np.array([X_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] for i in range(num_batches)])
Y_train_batches = np.array([Y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] for i in range(num_batches)])

print(f"Number of training batches: {len(X_train_batches)}")
print(f"X_train_batches shape: {X_train_batches.shape}")
print(f"Y_train_batches shape: {Y_train_batches.shape}")
