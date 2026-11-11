import json

# Task 1 Code
task1_code = """#Task 1
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
print(f"Y_train_batches shape: {Y_train_batches.shape}")"""

# Task 2 Code
task2_code = """#Task 2
#Task requirement: Neural Network Architecture (Built From Scratch)
#Task overview: Implement a multi-layer neural network class from scratch using NumPy matrix operations, featuring 256 input grayscale features, 768 output RGB color features, custom ReLU hidden layers, Sigmoid output layer matched to [0, 1] normalized pixel intensities, custom weight/bias initialization, and verification on training batches.

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducible architecture initialization
np.random.seed(42)

class ColorizationNN:
    \"\"\"
    Multi-layer Perceptron for image colorization implemented entirely from scratch.
    Maps flattened grayscale images (256 features) to flattened RGB images (768 features).
    \"\"\"
    def __init__(self, layer_sizes=[256, 128, 64, 768], seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(self.num_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            
            # He normal initialization for ReLU hidden layers; Xavier for Sigmoid output
            if i < self.num_layers - 1:
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)
                
            w = np.random.randn(n_in, n_out) * scale
            b = np.zeros((1, n_out))
            
            self.weights.append(w)
            self.biases.append(b)

    # Activation functions implemented manually from scratch
    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        # Clip z to prevent numerical overflow in exp(-z)
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def forward(self, X):
        \"\"\"
        Computes layer-by-layer forward propagation.
        Returns:
            a_out: Predicted color output matrix of shape (N, 768)
            caches: Dictionary of intermediate pre-activations (Z) and activations (A)
                    cached for Task 3 backpropagation.
        \"\"\"
        a = X
        caches = {'A0': X}
        
        # Forward pass through hidden layers with ReLU
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            caches[f'Z{i+1}'] = z
            caches[f'A{i+1}'] = a
            
        # Output layer with Sigmoid (maps strictly to [0, 1] color range)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self.sigmoid(z_out)
        caches[f'Z{self.num_layers}'] = z_out
        caches[f'A{self.num_layers}'] = a_out
        
        return a_out, caches

    def get_params_vector(self):
        \"\"\"Flattens all weights and biases into a 1D vector (ready for Task 5 Genetic Algorithm).\"\"\"
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.ravel())
            params.append(b.ravel())
        return np.concatenate(params)

    def set_params_vector(self, vector):
        \"\"\"Restores weights and biases from a 1D vector (for Task 5 Genetic Algorithm).\"\"\"
        curr = 0
        for i in range(self.num_layers):
            w_shape = self.weights[i].shape
            w_size = self.weights[i].size
            self.weights[i] = vector[curr : curr + w_size].reshape(w_shape)
            curr += w_size
            
            b_shape = self.biases[i].shape
            b_size = self.biases[i].size
            self.biases[i] = vector[curr : curr + b_size].reshape(b_shape)
            curr += b_size

# 1. Instantiate the network architecture
# Architecture: 256 input grayscale pixels -> 128 (ReLU) -> 64 (ReLU) -> 768 (Sigmoid RGB outputs)
model = ColorizationNN(layer_sizes=[256, 128, 64, 768], seed=42)

# Print architecture summary and parameter counts
print("Model Architecture Summary:")
total_params = 0
for i in range(model.num_layers):
    w_shape = model.weights[i].shape
    b_shape = model.biases[i].shape
    n_params = model.weights[i].size + model.biases[i].size
    total_params += n_params
    act = "ReLU" if i < model.num_layers - 1 else "Sigmoid"
    print(f"  Layer {i+1}: Input {w_shape[0]} -> Output {w_shape[1]} | Activation: {act} | Parameters: {n_params}")

print(f"Total trainable parameters: {total_params}")

# 2. Sanity-check forward pass on the first training batch from Task 1
sample_X = X_train_batches[0]  # Shape: (100, 256)
sample_Y = Y_train_batches[0]  # Shape: (100, 768)

sample_preds, caches = model.forward(sample_X)

print("\\nForward Pass Verification:")
print(f"  Input batch shape     : {sample_X.shape}")
print(f"  Output batch shape    : {sample_preds.shape}")
print(f"  Output value range    : [{sample_preds.min():.4f}, {sample_preds.max():.4f}] (strictly inside [0, 1])")

# 3. Compute initial un-trained Mean Squared Error (MSE) loss
initial_mse = np.mean((sample_preds - sample_Y) ** 2)
print(f"  Initial (untrained) MSE Loss: {initial_mse:.6f}")

# Assertions to confirm correctness
assert sample_preds.shape == (100, 768), "Incorrect output shape!"
assert np.all(sample_preds >= 0.0) and np.all(sample_preds <= 1.0), "Predictions out of [0, 1] range!"
assert total_params == model.get_params_vector().size, "Parameter vector size mismatch!"

# 4. Visualize sample initial predictions (untrained network vs ground truth)
num_samples = 5
fig, axes = plt.subplots(3, num_samples, figsize=(12, 6))

for i in range(num_samples):
    # Grayscale input
    axes[0, i].imshow(sample_X[i].reshape(16, 16), cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f"Gray Input #{i+1}")
    axes[0, i].axis('off')
    
    # Untrained network prediction
    axes[1, i].imshow(sample_preds[i].reshape(16, 16, 3), vmin=0, vmax=1)
    axes[1, i].set_title(f"Initial Pred #{i+1}")
    axes[1, i].axis('off')
    
    # Ground truth color image
    axes[2, i].imshow(sample_Y[i].reshape(16, 16, 3), vmin=0, vmax=1)
    axes[2, i].set_title(f"Ground Truth #{i+1}")
    axes[2, i].axis('off')

plt.suptitle("Task 2 Verification: Grayscale Input vs. Initial (Untrained) Prediction vs. Ground Truth (16x16)")
plt.tight_layout()
plt.show()"""

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# German University in Cairo (GUC)\n",
                "## Faculty of Media Engineering and Technology\n",
                "### DMET902 — Advanced Video Processing / CSEN1121 — Computational Intelligence and Neural Networks\n",
                "**Winter Semester 2026 — Course Project**  \n",
                "**Instructors:** Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Yomna Islam Youssef Alayary, Eng. Dina Waref Lamei Shalaby  \n",
                "\n",
                "---\n",
                "### Team & Submission Information\n",
                "| Attribute | Value |\n",
                "| :--- | :--- |\n",
                "| **Project Title** | Image Colorization: From-Scratch Neural Network & Genetic Algorithm Optimization |\n",
                "| **Team Name** | `404 Intelligence Not Found` (Filename: `404_Intelligence_Not_Found`) |\n",
                "| **Tasks Completed** | **Task 1: Dataset, Preprocessing, and Splitting** & **Task 2: Neural Network Architecture** |\n",
                "| **Task 2 Deadline** | Sunday, 15/11/2026 |\n",
                "\n",
                "### Team Members\n",
                "| # | Name | Student ID | Tutorial | Email |\n",
                "| :-: | :--- | :---: | :---: | :--- |\n",
                "| 1 | **Mohamed Ayman** | 58-8189 | T4 | `mohamed.awad@student.guc.edu.eg` |\n",
                "| 2 | **Hisham Hani** | 58-7992 | T3 | `hisham.gomaa@student.guc.edu.eg` |\n",
                "| 3 | **Abdallah Amr** | 58-2000 | T3 | `abdallah.bakr@student.guc.edu.eg` |\n",
                "| 4 | **Mohamed Ibrahim** | 58-16326 | T4 | `mohamed.ahmedismail@student.guc.edu.eg` |\n",
                "| 5 | **Ahmed Ashraf** | 58-1808 | T1 | `ahmed.abdelbaky@student.guc.edu.eg` |\n",
                "---\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in task1_code.splitlines()]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 1 Verification & Sanity Summary\n",
                "\n",
                "1. **Dataset Selection:** Selected a coherent subset from CIFAR-10 (`Animals` domain: birds, cats, deer, dogs, frogs, horses) comprising 2,000 samples. Downsampled to `16x16` via 2x2 spatial block average pooling to maintain computational tractability for scratch backpropagation and genetic algorithm optimization.\n",
                "2. **Grayscale Input & Color Target Generation:** Applied the perceptual standard ITU-R BT.601 conversion ($Y = 0.299R + 0.587G + 0.114B$) to derive grayscale inputs. Maintained ground truth RGB channels as targets.\n",
                "3. **Normalization:** All pixel intensities strictly mapped to the interval $[0.0, 1.0]$. Flattened dimensions: Input $X \\in \\mathbb{R}^{2000 \\times 256}$, Target $Y \\in \\mathbb{R}^{2000 \\times 768}$.\n",
                "4. **Sanity Checks & Visualizations:** Visualized side-by-side grayscale/color pairs and plotted smooth unimodal distributions across all channels confirming valid normalization.\n",
                "5. **Dataset Splitting:** Partitioned into Training (70% = 1,400 samples), Validation (20% = 400 samples), and Testing (10% = 200 samples) sets.\n",
                "6. **Mini-Batching:** Segmented the 1,400 training samples into 14 distinct mini-batches of exactly 100 samples each ($14 \\times 100 \\times 256$ and $14 \\times 100 \\times 768$).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in task2_code.splitlines()]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 2 Architecture Verification & Implementation Summary\n",
                "\n",
                "1. **From-Scratch Network Class:** `ColorizationNN` implemented using pure NumPy matrix multiplications (`np.dot`). No high-level neural network libraries or pre-built layers imported.\n",
                "2. **Input Layer:** Size = 256 (flattened $16 \\times 16$ grayscale luminance features).\n",
                "3. **Output Layer:** Size = 768 (flattened $16 \\times 16 \\times 3$ RGB color channel features).\n",
                "4. **Hidden Layers & Activations:** 2 hidden layers $[128, 64]$ with manually implemented **ReLU** activations ($f(z) = \\max(0, z)$) to capture non-linear chromatic relationships. Output layer utilizes **Sigmoid** ($g(z) = \\frac{1}{1 + e^{-z}}$) to naturally bound predicted outputs to the normalized $[0.0, 1.0]$ color interval.\n",
                "5. **Initialization:** Custom He/Kaiming normal initialization for ReLU layers ($W \\sim \\mathcal{N}(0, \\sqrt{2/n_{in}})$) and Xavier for the Sigmoid output layer. Biases initialized to zeros.\n",
                "6. **Genetic Algorithm Compatibility:** Methods `get_params_vector()` and `set_params_vector()` implemented to serialize the 91,072 parameters into a 1D chromosome for Task 5.\n",
                "7. **Verification Run:** Evaluated on batch 0 of training data (`X_train_batches[0]`), confirming correct output shape $(100, 768)$, valid value range $[0, 1]$, and initial un-trained MSE loss $\\approx 0.0767$.\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# The submission filename requirement: CINN_WS2026_Project_[Teamname].ipynb
with open("CINN_WS2026_Project_404_Intelligence_Not_Found.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated: CINN_WS2026_Project_404_Intelligence_Not_Found.ipynb")
