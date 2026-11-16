# DMET902 — Advanced Video Processing / CSEN1121
## Winter Semester 2026 — Course Project Report

**Faculty of Media Engineering and Technology**  
**German University in Cairo (GUC)**  

---

### Project & Submission Overview

* **Course:** DMET902 — Advanced Video Processing / CSEN1121 (Winter Semester 2026)
* **Instructors:** Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Yomna Islam Youssef Alayary, Eng. Dina Waref Lamei Shalaby
* **Project Title:** Image Colorization Using From-Scratch Neural Networks & Genetic Algorithm Optimization
* **Milestone:** **Task 1 (10 Marks)**, **Task 2 (15 Marks)** & **Task 3 — Backpropagation From Scratch (60 Marks)**
* **Task 3 Deadline:** Sunday, 29/11/2026
* **Team Name:** `404 Intelligence Not Found` (Archive: `CINN_WS2026_Project_404_Intelligence_Not_Found`)

---

### Team Members

| # | Student Name | Student ID | Tutorial Group | University Email |
| :-: | :--- | :---: | :---: | :--- |
| **1** | **Mohamed Ayman** | 58-8189 | T4 | `mohamed.awad@student.guc.edu.eg` |
| **2** | **Hisham Hani** | 58-7992 | T3 | `hisham.gomaa@student.guc.edu.eg` |
| **3** | **Abdallah Amr** | 58-2000 | T3 | `abdallah.bakr@student.guc.edu.eg` |
| **4** | **Mohamed Ibrahim** | 58-16326 | T4 | `mohamed.ahmedismail@student.guc.edu.eg` |
| **5** | **Ahmed Ashraf** | 58-1808 | T1 | `ahmed.abdelbaky@student.guc.edu.eg` |

---

# Part I: Task 1 — Dataset, Preprocessing, and Splitting

## 1. Executive Summary & Objective
The overarching objective of the course project is to build an **end-to-end image colorization engine** implemented entirely from scratch without high-level deep learning frameworks (e.g., PyTorch, Keras, TensorFlow). The system learns to map single-channel grayscale images to plausible 3-channel RGB color representations, first through analytical backpropagation (Task 3) and subsequently through evolutionary weight optimization via a Genetic Algorithm (Task 5).

Task 1 establishes the mathematical and structural data pipeline required for all downstream tasks. It guarantees that input data is cleanly preprocessed, appropriately sized, verified through statistical and visual sanity checks, and cleanly partitioned into training, validation, and testing mini-batches.

---

## 2. Dataset Selection & Domain Rationale

### Chosen Dataset: CIFAR-10 (`Animals` Domain)
* **Dataset:** CIFAR-10 (Canadian Institute for Advanced Research, 10 Classes)
* **Selected Classes:** **Animals** — Bird (`2`), Cat (`3`), Deer (`4`), Dog (`5`), Frog (`6`), Horse (`7`)
* **Total Curated Sample Size:** $N = 2{,}000$ images

### Theoretical & Empirical Justification:
1. **Domain Coherence & Preventing Desaturation:** Standard image colorization trained with Mean Squared Error (MSE) suffers from the known regression-to-the-mean phenomenon: if the model is fed radically dissimilar visual categories (e.g., metallic airplanes against blue skies mixed with green frogs and asphalt cars), the loss-minimizing output is a muddy, desaturated gray-brown average across all pixels. By focusing on a coherent semantic domain (**Animals** in natural environments), the distribution of plausible colors is tightly bounded to natural earth tones, vegetation greens, sky blues, and warm animal coats.
2. **Computational Tractability:** CIFAR-10 provides high-variance $32 \times 32$ color imagery. Selecting $N = 2{,}000$ samples provides sufficient statistical variety while allowing from-scratch vector operations, gradient propagation, and especially population-based Genetic Algorithm fitness iterations (Task 5) to execute within reasonable runtimes.

---

## 3. Dimensionality Reduction & Spatial Downsampling

To satisfy Task 1 Requirement 1 (*"Keep images small (e.g., 16×16 or 32×32) or work on small patches — a from-scratch network trains slowly"*), images were downsampled from $32 \times 32$ to **$16 \times 16$** using $2 \times 2$ spatial block average pooling:

$$\text{Downsampled}(i, j, c) = \frac{1}{4} \sum_{u=0}^{1} \sum_{v=0}^{1} \text{Original}(2i + u, 2j + v, c)$$

### Why $16 \times 16$ is the Optimal Design Choice:
* **Feature Dimensionality Comparison:**
  * **At $32 \times 32$:** Input vector length $= 32 \times 32 = 1{,}024$ grayscale features. Output vector length $= 32 \times 32 \times 3 = 3{,}072$ RGB features. A standard hidden layer with 128 neurons requires $(1{,}024 \times 128) + (128 \times 3{,}072) \approx 524{,}288$ trainable parameters.
  * **At $16 \times 16$:** Input vector length $= 16 \times 16 = 256$ grayscale features. Output vector length $= 16 \times 16 \times 3 = 768$ RGB features. Two hidden layers of $[128, 128]$ require $(256 \times 128) + 128 + (128 \times 128) + 128 + (128 \times 768) + 768 = 148{,}480$ parameters — a **$3.5\times$ reduction in computational complexity** compared to $32 \times 32$.
* **Impact on Task 5 (Genetic Algorithm):** Chromosomes representing 500k+ real-valued parameters across a population of 50 individuals suffer from severe curse-of-dimensionality degradation and memory bottlenecks. At $16 \times 16$ ($148{,}480$ weights and biases), evolutionary search and backpropagation converge reliably and efficiently.

---

## 4. Preprocessing & Feature Engineering

### 4.1 Pixel Intensity Normalization to $[0.0, 1.0]$
Raw integer pixel values $I_{\text{raw}} \in \{0, 1, \dots, 255\}$ are scaled to single-precision floating-point values:
$$I_{\text{norm}} = \frac{I_{\text{raw}}}{255.0}$$
This maps all input and target features strictly to the range $[0.0, 1.0]$, preventing gradient explosion and numerical overflow during backpropagation.

### 4.2 Grayscale Input Generation (ITU-R BT.601)
To generate realistic grayscale luminance inputs from the RGB ground truth, we apply the standard **ITU-R BT.601** perceptual luminance transform:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
This formula accounts for the physiological spectral sensitivity of human vision (greatest sensitivity to green wavelengths, moderate to red, and lowest to blue).

### 4.3 Matrix Feature Representation
* **Input Matrix ($X$):** $X \in \mathbb{R}^{2000 \times 256}$, containing flattened grayscale pixel intensities in $[0.0, 1.0]$.
* **Target Matrix ($Y$):** $Y \in \mathbb{R}^{2000 \times 768}$, containing flattened RGB color pixel values in $[0.0, 1.0]$.
* **Visual Tensors:** Original unflattened arrays $X_{\text{img}} \in \mathbb{R}^{2000 \times 16 \times 16}$ and $Y_{\text{img}} \in \mathbb{R}^{2000 \times 16 \times 16 \times 3}$ are preserved for visual rendering in Tasks 1, 2, 4, and 5.

---

## 5. Sanity Checks & Statistical Verification

To guarantee data integrity, automated programmatic assertions confirm that values remain bounded within $[0.0, 1.0]$:
```python
assert np.all(X_flat >= 0.0) and np.all(X_flat <= 1.0)
assert np.all(Y_flat >= 0.0) and np.all(Y_flat <= 1.0)
```

### Quantitative Distribution Statistics:
| Channel / Feature | Mean | Standard Deviation | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Grayscale Input ($Y$)** | **0.4643** | **0.2169** | **0.0000** | **1.0000** |
| **Red Channel ($R$)** | 0.4877 | 0.2274 | 0.0000 | 1.0000 |
| **Green Channel ($G$)** | 0.4656 | 0.2204 | 0.0000 | 1.0000 |
| **Blue Channel ($B$)** | 0.3959 | 0.2288 | 0.0000 | 1.0000 |

### Sanity Observations:
1. **Balanced Dynamic Range:** The grayscale distribution is centered at $\mu \approx 0.46$ with $\sigma \approx 0.22$, confirming that images exhibit balanced contrast without artificial clipping at black ($0.0$) or white ($1.0$).
2. **Channel Palette Properties:** The blue channel has a lower mean ($0.3959$) relative to red ($0.4877$) and green ($0.4656$), accurately reflecting natural terrestrial animal imagery dominated by warm earth tones, grass, and foliage rather than open sky or marine environments.
3. **Visual Verification:** Visualizing pairs of grayscale inputs versus ground-truth color targets confirms clean alignment without spatial shift or channel transposition.

---

## 6. Dataset Splitting & Mini-Batch Partitioning

In accordance with Task 1 Requirements 5 and 6:
* **Total Curated Samples:** $N = 2{,}000$
* **Training Set (70%):** $N_{\text{train}} = 1{,}400$ samples
  * Input: $X_{\text{train}} \in \mathbb{R}^{1400 \times 256}$
  * Target: $Y_{\text{train}} \in \mathbb{R}^{1400 \times 768}$
* **Validation Set (20%):** $N_{\text{val}} = 400$ samples
  * Input: $X_{\text{val}} \in \mathbb{R}^{400 \times 256}$
  * Target: $Y_{\text{val}} \in \mathbb{R}^{400 \times 768}$
* **Testing Set (10%):** $N_{\text{test}} = 200$ samples
  * Input: $X_{\text{test}} \in \mathbb{R}^{200 \times 256}$
  * Target: $Y_{\text{test}} \in \mathbb{R}^{200 \times 768}$

### Mini-Batch Formation:
The $1{,}400$ training samples are partitioned into **14 mini-batches of exactly 100 samples each**:
* **$X_{\text{train\_batches}}$:** Shape $(14, 100, 256)$
* **$Y_{\text{train\_batches}}$:** Shape $(14, 100, 768)$

This 3D tensor layout allows matrix multiplication in Task 2 and vectorized gradient descent in Task 3 without reshaping per batch.

---

# Part II: Task 2 — Neural Network Architecture (Built From Scratch)

## 7. Neural Network Architecture Design

In accordance with Task 2 specifications:
1. **Implementation Paradigm:** Pure object-oriented Python class (`NeuralNetwork`) using native NumPy array/matrix operations (`@` matrix multiplication). Pre-built layer or model classes (`torch.nn`, `keras.layers`, `sklearn`) are strictly prohibited and avoided.
2. **Input Layer Size:** $n_{\text{in}} = 256$ features, corresponding to the flattened $16 \times 16$ grayscale pixel intensities.
3. **Output Layer Size:** $n_{\text{out}} = 768$ features, corresponding to the flattened $16 \times 16 \times 3$ RGB color channel values.
4. **Hidden Layer Topology:** A balanced two-hidden-layer deep architecture $[128, 128]$ was selected:
   * **Hidden Layer 1:** 128 neurons with **ReLU** activation.
   * **Hidden Layer 2:** 128 neurons with **ReLU** activation.
   * **Output Layer:** 768 neurons with **Sigmoid** activation.

### Detailed Layer Dimension & Parameter Breakdown:
| Layer | Input Dimension | Output Dimension | Weight Matrix ($W$) | Bias Vector ($b$) | Activation | Trainable Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hidden Layer 1** | 256 | 128 | $(256, 128)$ | $(1, 128)$ | **ReLU** | $32{,}768 + 128 = \mathbf{32{,}896}$ |
| **Hidden Layer 2** | 128 | 128 | $(128, 128)$ | $(1, 128)$ | **ReLU** | $16{,}384 + 128 = \mathbf{16{,}512}$ |
| **Output Layer** | 128 | 768 | $(128, 768)$ | $(1, 768)$ | **Sigmoid** | $98{,}304 + 768 = \mathbf{99{,}072}$ |
| **Total** | — | — | — | — | — | **$\mathbf{148{,}480}$ parameters** |

---

## 8. Activation Functions Implemented From Scratch

### 8.1 Rectified Linear Unit (ReLU) & Derivative for Hidden Layers
For hidden layers, ReLU and its analytical derivative are implemented manually from first principles:
$$f(z) = \max(0, z), \qquad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```
* **Mathematical Justification:** ReLU avoids vanishing gradients across multiple hidden layers and computes with maximum vector efficiency in NumPy. The accompanying analytical derivative `relu_derivative(z)` pre-arms the engine for Task 3 Backpropagation.

### 8.2 Sigmoid & Derivative for the Output Layer
For the final color prediction layer, the Sigmoid function is implemented from scratch with numerical clipping to prevent `np.exp` overflow:
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
```python
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```
* **Mathematical Justification:** As mandated by Task 2 Requirement 4 (*"matched to your normalized color range"*), target RGB pixel values reside strictly in $[0.0, 1.0]$. The Sigmoid function naturally squashes pre-activations into $(0, 1)$. This guarantees every predicted pixel value is a mathematically valid normalized RGB color without requiring artificial post-hoc clamping.

---

## 9. Weight and Bias Initialization Strategy

Weights and biases are initialized from scratch without external library initializers:
1. **Hidden Layers (He / Kaiming Normal):**
   $$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$
   ```python
   scale = np.sqrt(2 / n_in)
   w = np.random.randn(n_in, n_out) * scale
   ```
   * Layer 1 ($n_{\text{in}} = 256$): $\sigma = \sqrt{2/256} = 0.0884$
   * Layer 2 ($n_{\text{in}} = 128$): $\sigma = \sqrt{2/128} = 0.1250$
   This maintains unit variance of activations across ReLU layers, preventing signals from vanishing to zero or exploding to infinity.
2. **Output Layer (Xavier / Glorot Normal):**
   $$W^{[3]} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{\text{in}}}}\right)$$
   ```python
   scale = np.sqrt(1 / n_in)
   w = np.random.randn(n_in, n_out) * scale
   ```
   * Layer 3 ($n_{\text{in}} = 128$): $\sigma = \sqrt{1/128} = 0.0884$, specifically tailored for logistic Sigmoid units.
3. **Biases:**
   Biases are initialized with small Gaussian perturbations scaled by $0.01$ to break initial symmetry across neurons:
   $$b^{[l]} \sim \mathcal{N}(0, 0.01^2)$$
   ```python
   b = np.random.randn(1, n_out) * 0.01
   ```

---

## 10. Forward Pass Propagation & Caching Mechanism

The forward pass is implemented via vectorized matrix multiplication (`@` operator):
$$Z^{[1]} = X W^{[1]} + b^{[1]}, \quad A^{[1]} = \text{ReLU}(Z^{[1]})$$
$$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}, \quad A^{[2]} = \text{ReLU}(Z^{[2]})$$
$$Z^{[3]} = A^{[2]} W^{[3]} + b^{[3]}, \quad \hat{Y} = A^{[3]} = \text{Sigmoid}(Z^{[3]})$$

### Forward Pass Cache:
To prepare seamlessly for **Task 3 (Backpropagation)**, all linear combinations ($Z^{[1]}, Z^{[2]}, Z^{[3]}$) and activation matrices ($A^{[0]}=X, A^{[1]}, A^{[2]}, A^{[3]}=\hat{Y}$) are stored directly on the network instance (`self.Z = [...]` and `self.A = [...]`) during every forward propagation.

---

## 11. Genetic Algorithm Chromosome Preparation (Task 5 Foresight)

The clean, modular design of the `NeuralNetwork` weight matrices ($W_1, W_2, W_3$) and bias vectors ($b_1, b_2, b_3$) enables straightforward flattening into a continuous 1D chromosome of length $148{,}480$:
* **Chromosome Encoding:** Concatenation of flattened weight and bias matrices.
* **Evolutionary Operators:** Uniform/arithmetic crossover and Gaussian mutation operate directly on the continuous chromosome space, and candidate solutions are mapped back into `network.weights` and `network.biases` for forward fitness evaluation against validation batches.

---

## 12. Verification & Experimental Sanity Results

The architecture was evaluated on batch 0 of the training set from Task 1 (`X_train_batches[0]`):
1. **Input Shape:** $(100, 256)$
2. **Predicted Output Shape:** $(100, 768)$ — perfectly matches $(N, 16 \times 16 \times 3)$.
3. **Dynamic Range:** Minimum predicted pixel intensity $= \mathbf{0.0478}$, Maximum $= \mathbf{0.9200}$. All predicted values reside strictly inside $[0.0, 1.0]$.
4. **Initial Un-Trained MSE Loss:**
   $$\text{MSE} = \frac{1}{N \cdot D} \sum_{i=1}^{N} \sum_{k=1}^{D} (\hat{Y}_{ik} - Y_{ik})^2 = \mathbf{0.071109}$$
   This matches the theoretical expected error of randomly initialized Sigmoid outputs centered around $0.5$ against naturally distributed pixel intensities.
5. **Visual Inspection:** Visualizing sample grayscale inputs, untrained network outputs, and ground-truth color targets confirms proper execution without spatial distortions. The untrained outputs render as a smooth, uniform tone characteristic of random initialization prior to gradient-based learning.

---

# Part III: Task 3 — Backpropagation, Implemented From Scratch (60 Marks)

## 14. Mathematical Derivation of Loss & Analytical Backpropagation

To adhere strictly to **Task 3** and **Regulation 8**, all derivatives and gradient descent parameter updates are derived mathematically and implemented using raw NumPy matrix arithmetic (`@`, `.T`, `np.sum`), without any automatic differentiation or pre-built optimizer frameworks.

### 14.1 Mean Squared Error (MSE) Loss
For a mini-batch of $m = 100$ samples and $D = 768$ RGB pixel outputs, the element-wise Mean Squared Error is:
$$\mathcal{L}(\hat{Y}, Y) = \frac{1}{m \cdot D} \sum_{i=1}^{m} \sum_{k=1}^{D} (\hat{Y}_{ik} - Y_{ik})^2$$

### 14.2 Chain Rule Gradient Propagation
Starting at the output predictions $\hat{Y} = A^{[3]}$, the analytical gradient propagates backward from Layer 3 to Layer 1:

1. **Output Layer ($l = 3$, Sigmoid):**
   $$\frac{\partial \mathcal{L}}{\partial A^{[3]}} = \frac{2}{m \cdot D} (A^{[3]} - Y)$$
   $$\delta^{[3]} = \frac{\partial \mathcal{L}}{\partial Z^{[3]}} = \frac{\partial \mathcal{L}}{\partial A^{[3]}} \odot \sigma'(Z^{[3]}), \quad \text{where } \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

2. **Parameter Gradients at Layer $l \in \{3, 2, 1\}$:**
   $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \left(A^{[l-1]}\right)^T \delta^{[l]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_{i=1}^{m} \delta^{[l]}_{i,:}$$

3. **Hidden Layer Backpropagation ($l \in \{2, 1\}$, ReLU):**
   $$\frac{\partial \mathcal{L}}{\partial A^{[l]}} = \delta^{[l+1]} \left(W^{[l+1]}\right)^T$$
   $$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} \odot \text{ReLU}'(Z^{[l]}), \quad \text{where } \text{ReLU}'(z) = \mathbb{I}(z > 0)$$

4. **Vanilla Gradient Descent Update:**
   $$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[l]}}, \qquad b^{[l]} \leftarrow b^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

---

## 15. Numerical Gradient Verification (Finite Differences)

To mathematically prove that the analytical backward pass (`backward_pass`) contains zero chain-rule or matrix-dimension errors, we verified analytical gradients against central finite differences ($\epsilon = 10^{-6}$):
$$\left.\frac{\partial \mathcal{L}}{\partial w_{rc}}\right|_{\text{num}} \approx \frac{\mathcal{L}(w_{rc} + \epsilon) - \mathcal{L}(w_{rc} - \epsilon)}{2\epsilon}$$

* **Maximum Relative Error across all 3 layers:** **`1.64e-07`** — confirming exact mathematical agreement between analytical backpropagation and numerical differentiation.

---

## 16. Hyperparameter Tuning & Learning Rate Sensitivity

Because element-wise MSE divides the squared-error sum by both batch size and output dimension ($m \times D = 100 \times 768 = 76{,}800$), each gradient entry is scaled by $1 / 76{,}800$. Starting around $\eta = 0.1$ yields very small updates over 40 epochs; therefore, we swept learning rates systematically:

| Learning Rate ($\eta$) | 40-Epoch Train MSE | 40-Epoch Val MSE | Convergence Behavior |
| :---: | :---: | :---: | :--- |
| `0.1` | 0.055204 | 0.055945 | Too slow (tiny steps due to $1/(m \cdot D)$ scaling) |
| `1.0` | 0.052978 | 0.053754 | Slow steady descent |
| `5.0` | 0.038663 | 0.039408 | Moderate descent |
| **`15.0` (Selected)** | **0.033454** | **0.034206** | **Smooth, stable, optimal convergence rate** |
| `30.0` | 0.037210 | 0.038503 | Begins oscillating / overshooting |
| `50.0` | 0.034824 | 0.035627 | High variance step oscillation |

Selecting **`LEARNING_RATE = 15.0`** (equivalent to an effective per-sample step size of $15.0 / 768 \approx 0.0195$ in sum-squared error) and training for **`EPOCHS = 300`** (far exceeding the 10-epoch minimum) produced smooth, non-divergent optimization.

---

## 17. Training Loop, Epoch Validation & Final Test Results

In strict accordance with Task 3 Requirement 5:
1. Each epoch iterates sequentially through all **14 mini-batches of 100 samples**, executing one forward pass, MSE computation, backward pass, and gradient descent update per batch (`4,200` total weight updates).
2. **Validation** (`X_val`, 400 samples) is evaluated **once at the end of every epoch**.
3. **Testing** (`X_test`, 200 samples) is executed **strictly once** after all 300 training epochs finish.

### Training & Validation Progression:
| Epoch | Training MSE | Validation MSE | Notes |
| :---: | :---: | :---: | :--- |
| **Initial (Untrained)** | — | **0.070611** | Random He/Xavier baseline |
| **Epoch 1** | 0.056027 | 0.055031 | Rapid initial bias/luminance alignment |
| **Epoch 25** | 0.034530 | 0.036599 | Strong structural learning |
| **Epoch 50** | 0.027759 | 0.026243 | Smooth loss descent |
| **Epoch 100** | 0.020476 | 0.020091 | >71% MSE reduction |
| **Epoch 150** | 0.017620 | 0.017813 | Generalization tracks training cleanly |
| **Epoch 200** | 0.016072 | 0.016346 | Continued refinement |
| **Epoch 250** | 0.014754 | 0.015437 | Fine chrominance tuning |
| **Epoch 300 (Final)** | **0.013692** | **0.014522** | **Converged (53.77s total runtime)** |
| **Final Test Pass (Once)** | — | — | **Test MSE = `0.015382` (78.2% reduction)** |

---

## 18. Summary Compliance Matrix (Tasks 1, 2 & 3)

| Regulation / Requirement | Course Specification | Implementation Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Regulation 1** | Team of up to 5 students | **Complete** | 5 students listed with IDs, tutorials, emails |
| **Regulation 2** | Zip file named `CINN_WS2026_Project_[Teamname].zip` | **Complete** | `CINN_WS2026_Project_404_Intelligence_Not_Found.zip` |
| **Regulations 3 & 4** | Report with operations, results, team details & grading notes | **Complete** | Sections 1–18 with full details |
| **Regulation 5** | Jupyter Notebook named identically to zip file | **Complete** | `CINN_WS2026_Project_404_Intelligence_Not_Found.ipynb` |
| **Regulation 6** | Every task collected in ONE single cell starting with #Task comments | **Complete** | Cell 2 (Task 1), Cell 5 (Task 2), Cell 8 (Task 3) |
| **Regulation 7** | Commented code explaining steps | **Complete** | Clean, student-style comments |
| **Regulation 8** | Pure NumPy arithmetic, no pre-built layers or autograd | **Complete** | Scratch matrix operations only |
| **Task 1.1 – 1.6** | Small dataset, Grayscale/RGB, Normalization, Sanity plots, 70/20/10 split, Batches of 100 | **Complete** | CIFAR-10 Animals, $16 \times 16$, 14 batches of 100 |
| **Task 2.1 – 2.5** | Scratch class, 256 in / 768 out, ReLU & Sigmoid + derivatives, He/Xavier init | **Complete** | `NeuralNetwork` class (148,480 params) |
| **Task 3.1 – 3.2** | Forward pass & MSE loss | **Complete** | `network.forward(X)` & `compute_mse_loss` |
| **Task 3.3** | Analytical backward pass via chain rule | **Complete** | `backward_pass` verified (`1.64e-07` error) |
| **Task 3.4** | Tuned gradient descent weight update | **Complete** | `update_weights` ($\eta = 15.0$) |
| **Task 3.5 – 3.6** | Per-batch updates, per-epoch validation, single final test pass, $\ge 10$ epochs | **Complete** | 300 epochs; Final Test MSE = `0.015382` |
