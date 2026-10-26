# CSEN1121 — Computational Intelligence and Neural Networks
## Winter Semester 2026 — Course Project Report
**German University in Cairo (GUC)**  
**Faculty of Media Engineering and Technology**  
**Course Instructors:** Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Dinah Waref Shalaby  

---

### Team & Submission Details

| Item | Details |
| :--- | :--- |
| **Project Title** | Image Colorization: From-Scratch Neural Network & Genetic Algorithm Optimization |
| **Team Name** | `404_Intelligence_Not_Found` |
| **Team Members** | Mohamed Ayman (and team members) |
| **Tutorial / Group** | `[e.g., CS-01]` |
| **Submission Deadline** | Task 1 Due: Friday, November 6, 2026 |

---

# Task 1: Dataset, Preprocessing, and Splitting (10 Marks)

## 1. Overview & Objective
Task 1 establishes the complete data foundation for the course project. The overall project objective is **image colorization**: training a neural network from scratch (and subsequently optimizing it via a genetic algorithm) to predict plausible color values from single-channel grayscale inputs.

To achieve successful colorization with a from-scratch shallow multilayer perceptron (MLP), the dataset must provide rich, realistic color information while remaining computationally tractable for matrix operations, analytical backpropagation, and population-based genetic search.

---

## 2. Dataset Selection & Domain Rationale
* **Dataset Choice:** **CIFAR-10** (Canadian Institute for Advanced Research 10-class dataset).
* **Selected Subdomain:** **Animals Domain** (Bird, Cat, Deer, Dog, Frog, Horse).
* **Sample Size:** $N = 2{,}000$ images.

### Justification:
1. **Domain Coherence:** CIFAR-10 contains 10 diverse classes spanning vehicles and living creatures. Training an unregularized, shallow MLP on disparate visual categories (e.g., airplanes in blue skies mixed with green frogs and metallic cars) forces the network to minimize MSE by predicting a desaturated gray-brown average across all semantic contexts. By constraining the dataset to an **Animals** domain, the network learns consistent semantic priors (e.g., natural foliage backgrounds, earth tones, skin/fur pigmentation).
2. **Computational Tractability:** CIFAR-10 provides standardized $32 \times 32$ color images with high variance. A curated sample size of $2{,}000$ instances ensures rapid iteration, exact mini-batch partitioning, and feasible fitness evaluation for the Task 5 Genetic Algorithm.

---

## 3. Image Downsampling & Dimensionality Analysis
In accordance with Task 1 Requirement 1 (*"Keep images small (e.g., 16×16 or 32×32) or work on small patches — a from-scratch network trains slowly"*), we downsampled the original $32 \times 32$ images to **$16 \times 16$** pixels using $2 \times 2$ spatial block average pooling.

$$\text{Processed}(i, j, c) = \frac{1}{4} \sum_{u=0}^{1} \sum_{v=0}^{1} \text{Original}(2i + u, 2j + v, c)$$

### Why $16 \times 16$ is Optimal:
* **Feature Dimensionality:**
  * At $32 \times 32$: Input vector length $= 1{,}024$ grayscale pixels; Output vector length $= 3{,}072$ RGB color values. A single hidden layer of $128$ neurons requires $(1024 \times 128) + (128 \times 3072) \approx 524{,}288$ trainable weights.
  * At $16 \times 16$: Input vector length $= 256$ grayscale pixels; Output vector length $= 768$ RGB color values. A hidden layer of $64$ neurons requires $(256 \times 64) + (64 \times 768) \approx 65{,}536$ weights.
* **Impact on Task 5 (Genetic Algorithm):** A chromosome representing $500{,}000+$ real-valued weights across a population of 50 individuals is computationally prohibitive and experiences severe convergence degradation (curse of dimensionality). In contrast, $\sim 65{,}000$ weights allow the genetic algorithm to evaluate fitness and converge in a reasonable timeframe.

---

## 4. Preprocessing & Normalization
1. **Pixel Intensity Normalization:**
   Raw CIFAR-10 integer bytes in the interval $[0, 255]$ are scaled to single-precision floating-point numbers in the interval $[0.0, 1.0]$:
   $$I_{\text{norm}} = \frac{I_{\text{raw}}}{255.0}$$
   This normalization ensures stable numerical gradients during backpropagation and prevents early saturation of downstream activation functions.

2. **Grayscale Input Generation:**
   To obtain the luminance input from the normalized RGB color target, we apply the standard **ITU-R BT.601** perceptual luminance transform:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
   This accounts for human trichromatic sensitivity (highest green sensitivity, moderate red sensitivity, lowest blue sensitivity).

3. **Feature Vector Representation:**
   * **Input Features ($X$):** Flattened grayscale image matrix $X \in \mathbb{R}^{N \times 256}$, with values in $[0.0, 1.0]$.
   * **Target Features ($Y$):** Flattened RGB color image matrix $Y \in \mathbb{R}^{N \times 768}$, with values in $[0.0, 1.0]$.
   * In addition to flattened representations, 2D/3D tensor representations $(N, 16, 16)$ and $(N, 16, 16, 3)$ are retained to support visual inspection in Tasks 4 and 5.

---

## 5. Sanity Checks & Statistical Verification
Strict programmatic assertions verify that all features strictly reside within $[0.0, 1.0]$:
```python
assert np.all(X_flattened >= 0.0) and np.all(X_flattened <= 1.0)
assert np.all(Y_flattened >= 0.0) and np.all(Y_flattened <= 1.0)
```

### Quantitative Distribution Statistics:
| Channel / Feature | Mean | Standard Deviation | Minimum | Maximum |
| :--- | :---: | :---: | :---: | :---: |
| **Grayscale Input ($Y$)** | **0.4643** | **0.2169** | **0.0000** | **1.0000** |
| **Red Channel ($R$)** | 0.4877 | 0.2274 | 0.0000 | 1.0000 |
| **Green Channel ($G$)** | 0.4656 | 0.2204 | 0.0000 | 1.0000 |
| **Blue Channel ($B$)** | 0.3959 | 0.2288 | 0.0000 | 1.0000 |

### Sanity Observations:
* The grayscale input distribution exhibits a well-balanced bell curve centered near $0.46$, showing neither black crushing nor white clipping.
* Channel means reflect the natural color palette of animals and outdoor environments (subdued blue tones averaging $0.3959$ compared to warm earth and foliage tones in red and green averaging $0.4877$ and $0.4656$).

---

## 6. Dataset Splitting & Batch Generation
In accordance with Task 1 Requirements 5 and 6:
* **Total Samples:** $N = 2{,}000$
* **Training Set (70%):** $N_{\text{train}} = 1{,}400$ samples $\rightarrow X_{\text{train}} \in \mathbb{R}^{1400 \times 256}, \; Y_{\text{train}} \in \mathbb{R}^{1400 \times 768}$
* **Validation Set (20%):** $N_{\text{val}} = 400$ samples $\rightarrow X_{\text{val}} \in \mathbb{R}^{400 \times 256}, \; Y_{\text{val}} \in \mathbb{R}^{400 \times 768}$
* **Testing Set (10%):** $N_{\text{test}} = 200$ samples $\rightarrow X_{\text{test}} \in \mathbb{R}^{200 \times 256}, \; Y_{\text{test}} \in \mathbb{R}^{200 \times 768}$
* **Mini-Batch Division:**
  The $1{,}400$ training instances are partitioned into exactly **14 mini-batches of 100 samples each**:
  * $X_{\text{train\_batches}} \in \mathbb{R}^{14 \times 100 \times 256}$
  * $Y_{\text{train\_batches}} \in \mathbb{R}^{14 \times 100 \times 768}$

---

## 7. Compliance Checklist with Course Regulations

| Regulation | Course Requirement | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Regulation 2** | Put all files in `.zip` named `CINN_WS2026_Project_[Teamname].zip` | Prepared | Zip script and directory ready |
| **Regulation 3 & 4** | Detailed project report with team metadata on cover | Prepared | Complete markdown report created |
| **Regulation 5** | Jupyter Notebook named identically to zip file | Prepared | `CINN_WS2026_Project_[Teamname].ipynb` generated |
| **Regulation 6** | Every task collected in **ONE single cell block** with `#Task number`, `#Task requirement`, `#Task overview` | Complete | Tested and fully verified in cell 1 |
| **Regulation 7** | Detailed step-by-step code comments | Complete | Every transformation step documented |
| **Regulation 8** | No plagiarism / no pre-built layers imported | Complete | Pure NumPy arithmetic, no external ML libraries |
