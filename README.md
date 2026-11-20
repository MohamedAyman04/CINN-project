# CSEN1121 — Computational Intelligence and Neural Networks (CINN)
## Winter Semester 2026 — Course Project

**Institution:** German University in Cairo (GUC)  
**Faculty:** Media Engineering and Technology  
**Instructors:** Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Yomna Islam Youssef Alayary, Eng. Dina Waref Lamei Shalaby  

---

## Project Overview
The objective of this course project is **Image Colorization**: training a neural network from scratch using custom matrix operations (NumPy) to predict RGB color values from single-channel grayscale inputs, followed by a comparative study using a Genetic Algorithm (GA) optimizer.

### Project Deadlines
* **Task 1:** Friday, 06/11/2026 (10 Marks) — *Dataset, Preprocessing, and Splitting*
* **Task 2:** Sunday, 15/11/2026 (15 Marks) — *Neural Network Architecture (From Scratch)*
* **Tasks 3, 4, 5:** Sunday, 29/11/2026 (105 Marks) — *Backpropagation, Evaluation, and Genetic Algorithm*

---

## Regulations Adherence
1. **Single Cell Execution (Regulation 6):** Every task is implemented inside **one single cell block** with the required comment format:
   ```python
   #Task number
   #Task requirement
   #Task overview
   ```
2. **From-Scratch Implementation (Regulation 8):** Pure NumPy arithmetic without high-level ML layers (e.g. PyTorch / Keras layers).
3. **Reproducibility:** Seed fixed at `RANDOM_SEED = 42`.
4. **Notebook & Zip Naming:** `CINN_WS2026_Project_[Teamname].ipynb` and `CINN_WS2026_Project_[Teamname].zip`.

---

## Task 1 Summary: Dataset, Preprocessing, and Splitting
* **Dataset:** CIFAR-10 (`Animals` domain: bird, cat, deer, dog, frog, horse) with 2,000 samples.
* **Resolution:** $16 \times 16$ pixels (downsampled via $2 \times 2$ block average pooling for computational tractability during from-scratch backprop and genetic algorithm optimization).
* **Grayscale Input:** ITU-R BT.601 formula: $Y = 0.299R + 0.587G + 0.114B$.
* **Target:** Ground truth RGB channels normalized to $[0.0, 1.0]$.
* **Splitting:**
  * Training: 70% (1,400 samples)
  * Validation: 20% (400 samples)
  * Testing: 10% (200 samples)
* **Mini-Batching:** Training data divided into 14 batches of exactly 100 samples each.

---

## Task 2 Summary: Neural Network Architecture (From Scratch)
* **Class:** `NeuralNetwork` implemented in pure NumPy matrix arithmetic (`@`).
* **Topology:** $256 \to 128 \text{ (ReLU)} \to 128 \text{ (ReLU)} \to 768 \text{ (Sigmoid)}$ (**148,480** trainable parameters).
* **Initialization:** He normal for ReLU hidden layers, Xavier normal for Sigmoid output layer, and small Gaussian biases ($\sigma = 0.01$).

---

## Task 3 Summary: Backpropagation (From Scratch)
* **Loss:** Element-wise Mean Squared Error (MSE) $\mathcal{L} = \frac{1}{m \cdot D}\sum (\hat{Y} - Y)^2$.
* **Gradient Verification:** Analytical chain-rule backpropagation verified against central finite differences (`1.64e-07` max relative error).
* **Training:** 300 epochs of mini-batch gradient descent (`LEARNING_RATE = 15.0`) with validation evaluated after each epoch and a single test evaluation pass at the end.
* **Results:** Validation MSE decreased from `0.070611` to `0.014522`; Final Test MSE = `0.015382`.

---

## Task 4 Summary: Evaluation and Results
* **Final Test Metrics (200 Unseen Images):** MSE = `0.015382`, RMSE = `0.124022`, MAE = `0.094342`, PSNR = `18.13 dB` (Per-channel MSE: R `0.014842`, G `0.011189`, B `0.020114`).
* **Visual & Hyperparameter Analysis:** Displayed 10 side-by-side colorizations (`task4_test_colorizations.png`), evaluated Learning Rate, Batch Size, and Hidden Width sensitivity, and analyzed MSE desaturation / bottleneck smoothing limitations.

---

## Task 5 Summary: Genetic Algorithm Comparison
* **GA Formulation:** Same `[256, 128, 128, 768]` architecture (`148,480` float32 chromosome) with Tournament Selection ($k=3$), Hybrid Uniform/Blend Crossover, Annealing Gaussian Mutation ($\sigma: 0.02 \to 0.005$), and Elitist Replacement (`2` elites, `20` pop, `100` gens).
* **Comparison:** Backpropagation reached `0.015382` Test MSE in `54.10s`, outperforming the GA (`0.055276` Test MSE in `36.05s`) by `3.6×` on accuracy because exact first-order gradients overcome the 148,480-dimensional search bottleneck.

---

## Running in Kaggle
1. Open or upload `CINN_WS2026_Project_404_Intelligence_Not_Found.ipynb` in Kaggle.
2. In the top right corner of the Kaggle notebook, click **`+ Add Input`**.
3. Search for **`cifar-10-python`** and click **Add** (the dataset mounts to `/kaggle/input/...`).
4. Run all 5 task cells sequentially (Tasks 1–5).
