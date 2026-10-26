# CSEN1121 — Computational Intelligence and Neural Networks (CINN)
## Winter Semester 2026 — Course Project

**Institution:** German University in Cairo (GUC)  
**Faculty:** Media Engineering and Technology  
**Instructors:** Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Dinah Waref Shalaby  

---

## Project Overview
The objective of this course project is **Image Colorization**: training a neural network from scratch using custom matrix operations (NumPy) to predict RGB color values from single-channel grayscale inputs, followed by a comparative study using a Genetic Algorithm (GA) optimizer.

### Project Deadlines
* **Task 1:** Friday, 06/11/2026 (10 Marks) — *Dataset, Preprocessing, and Splitting*
* **Task 2:** Thursday, 15/11/2026 (15 Marks) — *Neural Network Architecture (From Scratch)*
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

## Running in Kaggle
1. Open or upload `CINN_WS2026_Project_404_Intelligence_Not_Found.ipynb` in Kaggle.
2. In the top right corner of the Kaggle notebook, click **`+ Add Input`**.
3. Search for **`cifar-10-python`** and click **Add** (the dataset mounts to `/kaggle/input/...`).
4. Run the Task 1 cell block. It directly accesses the Kaggle input without downloading anything, performs the preprocessing, plots distributions inline, splits, and batches the data ready for Task 2.
