import json

# Read task1.py source code
with open("task1.py", "r") as f:
    task1_code = f.read()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CSEN1121 — Computational Intelligence and Neural Networks\n",
                "## Winter Semester 2026 — Course Project\n",
                "**German University in Cairo (GUC)**  \n",
                "**Instructors:** Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Dinah Waref Shalaby  \n",
                "\n",
                "---\n",
                "### Team & Submission Information\n",
                "| Attribute | Value |\n",
                "| :--- | :--- |\n",
                "| **Team Name** | `404_Intelligence_Not_Found` |\n",
                "| **Team Members** | Mohamed Ayman (and team members) |\n",
                "| **Tutorial / Group** | `[Your Tutorial Group]` |\n",
                "| **Current Task** | **Task 1: Dataset, Preprocessing, and Splitting (10 Marks)** |\n",
                "| **Deadline** | Friday, November 6, 2026 |\n",
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
