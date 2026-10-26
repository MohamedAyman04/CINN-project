#Task 1
#Task requirement: Dataset, Preprocessing, and Splitting
#Task overview: Load and preprocess CIFAR-10 color images, generate normalized grayscale inputs [0, 1] and color targets [0, 1], perform data sanity checks via visualization and distribution plots, split into 70% training, 20% validation, and 10% testing sets, and divide training data into batches of 100 samples each.

import os
import glob
import pickle
import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Configuration & Reproducibility
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Hyperparameters for dataset preparation
# Using 16x16 downsampling: keeps the model lightweight (~65k parameters)
# so from-scratch backpropagation and Task 5 (Genetic Algorithm) remain computationally feasible.
TARGET_SIZE = (16, 16)
TOTAL_SAMPLES = 2000        # 70% = 1400 (14 batches of 100), 20% = 400, 10% = 200
BATCH_SIZE = 100            # Specified in Requirement 6

print("================================================================================")
print("CSEN1121 CINN Project — Task 1: Dataset, Preprocessing, and Splitting")
print("================================================================================")

# ==============================================================================
# 2. Dataset Discovery (Handles Kaggle Upload, Kaggle Dataset, Local & Auto-download)
# ==============================================================================
# 2a. Check if extracted batches.meta already exists anywhere in input or workspace
candidate_meta = (
    glob.glob("/kaggle/input/**/batches.meta", recursive=True) +
    glob.glob("../input/**/batches.meta", recursive=True) +
    glob.glob("./**/batches.meta", recursive=True)
)

cifar_dir = None
if candidate_meta:
    cifar_dir = os.path.dirname(candidate_meta[0])
    print(f"\n[Step 1] Found extracted CIFAR-10 directory at: {cifar_dir}")
else:
    # 2b. Check if user uploaded cifar-10-python.tar.gz directly to Kaggle input
    tar_candidates = (
        glob.glob("/kaggle/input/**/cifar-10-python.tar.gz", recursive=True) +
        glob.glob("/kaggle/input/**/*.tar.gz", recursive=True) +
        glob.glob("../input/**/*.tar.gz", recursive=True) +
        glob.glob("./cifar_data/*.tar.gz")
    )
    
    os.makedirs("cifar_data", exist_ok=True)
    if tar_candidates:
        uploaded_tar = tar_candidates[0]
        print(f"\n[Step 1] Detected uploaded archive at: {uploaded_tar}")
        print("[Step 1] Extracting dataset archive...")
        with tarfile.open(uploaded_tar, "r:gz") as tar:
            tar.extractall("cifar_data")
        cifar_dir = "cifar_data/cifar-10-batches-py"
    else:
        # 2c. Fallback: Auto-download from official URL if internet is enabled
        print("\n[Step 1] Archive not found in input. Downloading from official repository...")
        tar_path = "cifar_data/cifar-10-python.tar.gz"
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(cifar_url, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall("cifar_data")
        cifar_dir = "cifar_data/cifar-10-batches-py"

print(f"[Step 1] CIFAR-10 ready at: {cifar_dir}")

# Read metadata for label names
meta_file = os.path.join(cifar_dir, "batches.meta")
with open(meta_file, "rb") as f:
    meta = pickle.load(f, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

print(f"Dataset classes available: {class_names}")

# Filter for the 'Animals' domain (Requirement 1: cohesive classification domain):
# 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse
SELECTED_CLASSES = [2, 3, 4, 5, 6, 7]
selected_names = [class_names[c] for c in SELECTED_CLASSES]
print(f"Selected domain: Animals ({', '.join(selected_names)})")

raw_data_list = []
raw_label_list = []

for b_idx in range(1, 6):
    batch_path = os.path.join(cifar_dir, f"data_batch_{b_idx}")
    with open(batch_path, "rb") as f:
        batch_dict = pickle.load(f, encoding="bytes")
        b_data = batch_dict[b"data"]
        b_labels = np.array(batch_dict[b"labels"])
        
        mask = np.isin(b_labels, SELECTED_CLASSES)
        raw_data_list.append(b_data[mask])
        raw_label_list.append(b_labels[mask])

all_raw_data = np.concatenate(raw_data_list, axis=0)
all_raw_labels = np.concatenate(raw_label_list, axis=0)

# Reshape raw data (N, 3072) into standard format (N, 32, 32, 3)
all_images_32 = all_raw_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Randomly sample TOTAL_SAMPLES images
sample_indices = np.random.permutation(len(all_images_32))[:TOTAL_SAMPLES]
sampled_images = all_images_32[sample_indices]
sampled_labels = all_raw_labels[sample_indices]
print(f"Sampled {TOTAL_SAMPLES} images for processing.")

# ==============================================================================
# 3. Downsampling (16x16) & Normalization to [0, 1]
# ==============================================================================
# Downsampling 32x32 to 16x16 via 2x2 spatial average pooling
if TARGET_SIZE == (16, 16):
    processed_images = sampled_images.reshape(-1, 16, 2, 16, 2, 3).mean(axis=(2, 4))
else:
    processed_images = sampled_images.astype(np.float32)

# Normalize color pixel values to [0.0, 1.0] range
color_targets = processed_images.astype(np.float32) / 255.0

# ==============================================================================
# 4. Generate Grayscale Input & Color Target
# ==============================================================================
# Standard ITU-R BT.601 conversion: Y = 0.299*R + 0.587*G + 0.114*B
grayscale_inputs = (
    0.299 * color_targets[:, :, :, 0] +
    0.587 * color_targets[:, :, :, 1] +
    0.114 * color_targets[:, :, :, 2]
)

# Flatten feature vectors for neural network layers:
# Input: (N, 256) flattened grayscale values
# Target: (N, 768) flattened RGB color values
X_flattened = grayscale_inputs.reshape(TOTAL_SAMPLES, -1)
Y_flattened = color_targets.reshape(TOTAL_SAMPLES, -1)

# Assertions to sanity-check normalization and shapes
assert np.all(X_flattened >= 0.0) and np.all(X_flattened <= 1.0), "Grayscale pixel values out of [0, 1]!"
assert np.all(Y_flattened >= 0.0) and np.all(Y_flattened <= 1.0), "Color pixel values out of [0, 1]!"
assert X_flattened.shape == (TOTAL_SAMPLES, TARGET_SIZE[0] * TARGET_SIZE[1]), "Incorrect input shape!"
assert Y_flattened.shape == (TOTAL_SAMPLES, TARGET_SIZE[0] * TARGET_SIZE[1] * 3), "Incorrect target shape!"

print(f"\n[Step 2 & 3] Preprocessing and Normalization Verified:")
print(f"  • Grayscale Input X  : shape = {X_flattened.shape}, range = [{X_flattened.min():.4f}, {X_flattened.max():.4f}]")
print(f"  • Color Target Y     : shape = {Y_flattened.shape}, range = [{Y_flattened.min():.4f}, {Y_flattened.max():.4f}]")

# ==============================================================================
# 5. Sanity-Check Visualizations & Distribution Analysis
# ==============================================================================
print("\n[Step 4] Generating Sanity-Check Visualizations and Plots...")

# Visualization 1: Grayscale Input vs Color Target Pairs
num_vis = 6
fig_pairs, axes_pairs = plt.subplots(2, num_vis, figsize=(15, 5))

for i in range(num_vis):
    lbl = class_names[sampled_labels[i]]
    # Grayscale Input
    axes_pairs[0, i].imshow(grayscale_inputs[i], cmap="gray", vmin=0, vmax=1)
    axes_pairs[0, i].set_title(f"Gray Input #{i+1}\n({lbl})", fontsize=10, fontweight="bold")
    axes_pairs[0, i].axis("off")
    
    # Color Target
    axes_pairs[1, i].imshow(color_targets[i], vmin=0, vmax=1)
    axes_pairs[1, i].set_title(f"Color Target #{i+1}\n({lbl})", fontsize=10, fontweight="bold")
    axes_pairs[1, i].axis("off")

plt.suptitle(f"Task 1 Sanity Check: Sample Grayscale Inputs vs. Color Targets ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()

# Visualization 2: Pixel-Value Distribution Histograms
fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 5))

# Grayscale distribution
axes_dist[0].hist(grayscale_inputs.flatten(), bins=50, color="#4A5568", alpha=0.85, edgecolor="black", density=True)
axes_dist[0].set_title("Pixel Value Distribution: Grayscale Input [0, 1]", fontsize=11, fontweight="bold")
axes_dist[0].set_xlabel("Normalized Intensity")
axes_dist[0].set_ylabel("Probability Density")
axes_dist[0].grid(True, linestyle="--", alpha=0.5)

# RGB distributions
axes_dist[1].hist(color_targets[:, :, :, 0].flatten(), bins=50, color="#E53E3E", alpha=0.5, edgecolor="darkred", density=True, label="Red Channel")
axes_dist[1].hist(color_targets[:, :, :, 1].flatten(), bins=50, color="#38A169", alpha=0.5, edgecolor="darkgreen", density=True, label="Green Channel")
axes_dist[1].hist(color_targets[:, :, :, 2].flatten(), bins=50, color="#3182CE", alpha=0.5, edgecolor="darkblue", density=True, label="Blue Channel")
axes_dist[1].set_title("Pixel Value Distribution: RGB Color Channels [0, 1]", fontsize=11, fontweight="bold")
axes_dist[1].set_xlabel("Normalized Intensity")
axes_dist[1].set_ylabel("Probability Density")
axes_dist[1].legend(loc="upper right", frameon=True)
axes_dist[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Quantitative Distribution Summary
print("\n--- Quantitative Pixel-Value Distribution Statistics ---")
print(f"  • Grayscale : Mean = {grayscale_inputs.mean():.4f}, Std = {grayscale_inputs.std():.4f}, Min = {grayscale_inputs.min():.4f}, Max = {grayscale_inputs.max():.4f}")
print(f"  • Red Chan  : Mean = {color_targets[:,:,:,0].mean():.4f}, Std = {color_targets[:,:,:,0].std():.4f}, Min = {color_targets[:,:,:,0].min():.4f}, Max = {color_targets[:,:,:,0].max():.4f}")
print(f"  • Green Chan: Mean = {color_targets[:,:,:,1].mean():.4f}, Std = {color_targets[:,:,:,1].std():.4f}, Min = {color_targets[:,:,:,1].min():.4f}, Max = {color_targets[:,:,:,1].max():.4f}")
print(f"  • Blue Chan : Mean = {color_targets[:,:,:,2].mean():.4f}, Std = {color_targets[:,:,:,2].std():.4f}, Min = {color_targets[:,:,:,2].min():.4f}, Max = {color_targets[:,:,:,2].max():.4f}")

# ==============================================================================
# 6. Dataset Splitting: Training (70%), Validation (20%), Testing (10%)
# ==============================================================================
n_train = int(0.70 * TOTAL_SAMPLES)   # 1400 samples
n_val = int(0.20 * TOTAL_SAMPLES)     # 400 samples
n_test = TOTAL_SAMPLES - n_train - n_val  # 200 samples

# Reproducible shuffle split
indices_split = np.random.permutation(TOTAL_SAMPLES)
train_idx = indices_split[:n_train]
val_idx = indices_split[n_train:n_train + n_val]
test_idx = indices_split[n_train + n_val:]

# Flattened arrays for model operations
X_train, Y_train = X_flattened[train_idx], Y_flattened[train_idx]
X_val, Y_val = X_flattened[val_idx], Y_flattened[val_idx]
X_test, Y_test = X_flattened[test_idx], Y_flattened[test_idx]

# Unflattened arrays preserved for visual inspection in Tasks 4 & 5
gray_train, color_train = grayscale_inputs[train_idx], color_targets[train_idx]
gray_val, color_val = grayscale_inputs[val_idx], color_targets[val_idx]
gray_test, color_test = grayscale_inputs[test_idx], color_targets[test_idx]

print("\n[Step 5] Dataset Split Completed (70% / 20% / 10%):")
print(f"  • Training set   : {X_train.shape[0]} samples ({X_train.shape[0]/TOTAL_SAMPLES*100:.1f}%) | X: {X_train.shape}, Y: {Y_train.shape}")
print(f"  • Validation set : {X_val.shape[0]} samples ({X_val.shape[0]/TOTAL_SAMPLES*100:.1f}%) | X: {X_val.shape}, Y: {Y_val.shape}")
print(f"  • Testing set    : {X_test.shape[0]} samples ({X_test.shape[0]/TOTAL_SAMPLES*100:.1f}%) | X: {X_test.shape}, Y: {Y_test.shape}")

# ==============================================================================
# 7. Dividing Training Data into Batches of 100 Samples
# ==============================================================================
num_train_batches = len(X_train) // BATCH_SIZE

# Create list of batches
X_train_batches = np.array([X_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] for i in range(num_train_batches)])
Y_train_batches = np.array([Y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] for i in range(num_train_batches)])

print("\n[Step 6] Dividing Training Data into Batches of 100:")
print(f"  • Total training mini-batches : {len(X_train_batches)}")
print(f"  • Batch array shape for X     : {X_train_batches.shape} (Batches x Batch_Size x Features)")
print(f"  • Batch array shape for Y     : {Y_train_batches.shape} (Batches x Batch_Size x Targets)")
print(f"  • Every batch contains exactly {BATCH_SIZE} samples.")

print("\n================================================================================")
print(" Task 1 Execution Completed Successfully!")
print("================================================================================")
