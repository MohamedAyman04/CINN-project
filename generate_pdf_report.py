import os
import base64
import weasyprint

def get_b64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

img_pairs_b64 = get_b64_image("task1_sanity_check_pairs.png")
img_dist_b64 = get_b64_image("task1_pixel_distributions.png")
img_task2_b64 = get_b64_image("task2_initial_predictions.png")
img_task3_curve_b64 = get_b64_image("task3_training_curve.png")
img_task3_preview_b64 = get_b64_image("task3_trained_preview.png")
img_task4_curves_b64 = get_b64_image("task4_learning_curves.png")
img_task4_colors_b64 = get_b64_image("task4_test_colorizations.png")
img_task4_hp_b64 = get_b64_image("task4_hyperparameter_study.png")
img_task5_cmp_b64 = get_b64_image("task5_ga_vs_bp_comparison.png")
img_task5_vis_b64 = get_b64_image("task5_visual_comparison.png")

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CINN Course Project Report — Tasks 1–5 (Complete Submission)</title>
<style>
    @page {{
        size: A4;
        margin: 14mm 14mm 14mm 14mm;
        @bottom-right {{
            content: "Page " counter(page) " of " counter(pages);
            font-size: 8pt;
            color: #718096;
        }}
        @bottom-left {{
            content: "DMET902 / CSEN1121 — Course Project Report (Winter 2026)";
            font-size: 8pt;
            color: #718096;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 9.3pt;
        line-height: 1.4;
        color: #2D3748;
        margin: 0;
    }}
    .header {{
        text-align: center;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 10px;
        margin-bottom: 14px;
    }}
    .header h1 {{
        font-size: 16pt;
        margin: 0 0 3px 0;
        color: #1A202C;
    }}
    .header h2 {{
        font-size: 12pt;
        font-weight: 600;
        margin: 0 0 4px 0;
        color: #4A5568;
    }}
    .header p {{
        font-size: 9pt;
        margin: 2px 0;
        color: #718096;
    }}
    .meta-box {{
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }}
    .meta-box table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .meta-box td {{
        padding: 3px 6px;
        font-size: 9pt;
    }}
    .meta-box td.label {{
        font-weight: 600;
        color: #4A5568;
        width: 25%;
    }}
    h1.part-title {{
        font-size: 13.5pt;
        color: #1A365D;
        background: #EBF8FF;
        border-left: 4px solid #3182CE;
        padding: 5px 10px;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }}
    h2.section-title {{
        font-size: 11.5pt;
        color: #2B6CB0;
        border-bottom: 1.5px solid #E2E8F0;
        padding-bottom: 3px;
        margin-top: 14px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}
    h3.subsection-title {{
        font-size: 10pt;
        color: #2D3748;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}
    table.data-table th, table.data-table td {{
        border: 1px solid #CBD5E0;
        padding: 5px 8px;
        text-align: left;
    }}
    table.data-table th {{
        background-color: #EDF2F7;
        color: #2D3748;
        font-weight: 600;
    }}
    table.data-table tr:nth-child(even) {{
        background-color: #F7FAFC;
    }}
    .center-text {{
        text-align: center;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 5px;
        font-size: 7.5pt;
        font-weight: 600;
        background: #C6F6D5;
        color: #22543D;
        border-radius: 4px;
    }}
    .img-container {{
        text-align: center;
        margin: 10px 0;
        page-break-inside: avoid;
    }}
    .img-container img {{
        max-width: 92%;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
    }}
    .img-caption {{
        font-size: 8pt;
        color: #718096;
        margin-top: 3px;
        font-style: italic;
    }}
    code {{
        font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 8pt;
        background: #EDF2F7;
        padding: 1px 4px;
        border-radius: 3px;
        color: #C53030;
    }}
    pre {{
        background: #2D3748;
        color: #F7FAFC;
        padding: 8px;
        border-radius: 4px;
        font-size: 8pt;
        overflow-x: auto;
        page-break-inside: avoid;
    }}
    .page-break {{
        page-break-before: always;
    }}
</style>
</head>
<body>

<div class="header">
    <p style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #4A5568;">German University in Cairo (GUC) • Faculty of Media Engineering & Technology</p>
    <h1>DMET902 — Advanced Video Processing / CSEN1121</h1>
    <h2>CINN Project — Winter Semester 2026</h2>
    <p>Instructors: Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Yomna Islam Youssef Alayary, Eng. Dina Waref Lamei Shalaby</p>
</div>

<div class="meta-box">
    <table>
        <tr>
            <td class="label">Project Title:</td>
            <td><strong>Image Colorization Using From-Scratch Neural Networks & Genetic Algorithm</strong></td>
        </tr>
        <tr>
            <td class="label">Milestone Scope:</td>
            <td><strong>Complete Project Submission — Tasks 1–5 (150 Marks Total)</strong></td>
        </tr>
        <tr>
            <td class="label">Final Deadline:</td>
            <td>Sunday, November 29, 2026</td>
        </tr>
        <tr>
            <td class="label">Team Name:</td>
            <td><code>404 Intelligence Not Found</code> (Archive: <code>CINN_WS2026_Project_404_Intelligence_Not_Found</code>)</td>
        </tr>
    </table>
</div>

<h3 class="subsection-title" style="margin-top: 10px;">Team Members</h3>
<table class="data-table">
    <thead>
        <tr>
            <th class="center-text" style="width: 5%;">#</th>
            <th style="width: 25%;">Student Name</th>
            <th class="center-text" style="width: 15%;">Student ID</th>
            <th class="center-text" style="width: 12%;">Tutorial</th>
            <th style="width: 43%;">University Email</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="center-text">1</td>
            <td><strong>Mohamed Ayman</strong></td>
            <td class="center-text">58-8189</td>
            <td class="center-text">T4</td>
            <td><code>mohamed.awad@student.guc.edu.eg</code></td>
        </tr>
        <tr>
            <td class="center-text">2</td>
            <td><strong>Hisham Hani</strong></td>
            <td class="center-text">58-7992</td>
            <td class="center-text">T3</td>
            <td><code>hisham.gomaa@student.guc.edu.eg</code></td>
        </tr>
        <tr>
            <td class="center-text">3</td>
            <td><strong>Abdallah Amr</strong></td>
            <td class="center-text">58-2000</td>
            <td class="center-text">T3</td>
            <td><code>abdallah.bakr@student.guc.edu.eg</code></td>
        </tr>
        <tr>
            <td class="center-text">4</td>
            <td><strong>Mohamed Ibrahim</strong></td>
            <td class="center-text">58-16326</td>
            <td class="center-text">T4</td>
            <td><code>mohamed.ahmedismail@student.guc.edu.eg</code></td>
        </tr>
        <tr>
            <td class="center-text">5</td>
            <td><strong>Ahmed Ashraf</strong></td>
            <td class="center-text">58-1808</td>
            <td class="center-text">T1</td>
            <td><code>ahmed.abdelbaky@student.guc.edu.eg</code></td>
        </tr>
    </tbody>
</table>

<h1 class="part-title">Part I: Task 1 — Dataset, Preprocessing, and Splitting</h1>

<h2 class="section-title">1. Executive Summary & Objective</h2>
<p>
The objective of the course project is to build an end-to-end image colorization engine implemented entirely from scratch without high-level deep learning frameworks (e.g. PyTorch, Keras, TensorFlow). The system learns to map single-channel grayscale images to plausible 3-channel RGB color representations, first through analytical backpropagation (Task 3) and subsequently through evolutionary weight optimization via a Genetic Algorithm (Task 5).
</p>

<h2 class="section-title">2. Dataset Selection & Domain Rationale</h2>
<p>
<strong>Selected Dataset:</strong> CIFAR-10 (Canadian Institute for Advanced Research, 10 Classes)<br>
<strong>Target Subdomain:</strong> <em>Animals Domain</em> — Bird (<code>2</code>), Cat (<code>3</code>), Deer (<code>4</code>), Dog (<code>5</code>), Frog (<code>6</code>), Horse (<code>7</code>)<br>
<strong>Curated Sample Count:</strong> N = 2,000 images
</p>
<p>
<strong>Theoretical Justification:</strong> Standard image colorization trained with Mean Squared Error (MSE) suffers from the regression-to-the-mean phenomenon: if the model is fed radically dissimilar visual categories (e.g., airplanes in blue sky mixed with green frogs and asphalt cars), the loss-minimizing prediction is a desaturated gray-brown average across all pixels. By focusing on a coherent semantic domain (Animals in natural environments), the distribution of plausible colors is tightly bounded to natural earth tones, vegetation greens, sky blues, and warm animal coats. Furthermore, 2,000 samples provides sufficient statistical variety while allowing from-scratch vector operations, gradient propagation, and population-based Genetic Algorithm fitness iterations (Task 5) to execute within reasonable runtimes.
</p>

<h2 class="section-title">3. Spatial Downsampling & Dimensionality Analysis</h2>
<p>
In accordance with Task 1 Requirement 1 (<em>"Keep images small (e.g., 16×16 or 32×32) or work on small patches — a from-scratch network trains slowly"</em>), images were downsampled from 32×32 to <strong>16×16</strong> using 2×2 spatial block average pooling:
</p>
<p style="text-align: center; font-style: italic; background: #F7FAFC; padding: 4px; border-radius: 4px; border: 1px solid #E2E8F0; margin: 4px 0;">
Downsampled(i, j, c) = &frac14; &sum;<sub>u=0</sub><sup>1</sup> &sum;<sub>v=0</sub><sup>1</sup> Original(2i + u, 2j + v, c)
</p>
<ul>
    <li><strong>Feature Dimensionality at 32×32:</strong> Input = 1,024 grayscale features; Output = 3,072 RGB features. A single hidden layer of 128 neurons requires (1,024 × 128) + (128 × 3,072) &approx; <strong>524,288 weights</strong>.</li>
    <li><strong>Feature Dimensionality at 16×16:</strong> Input = 256 grayscale features; Output = 768 RGB features. Two hidden layers of [128, 128] require <strong>148,480 parameters</strong> — a <strong>3.5&times; reduction in computational complexity</strong> compared to 32×32.</li>
    <li><strong>Critical for Task 5 (Genetic Algorithm):</strong> Chromosomes representing 500k+ real-valued parameters across a population of 50 individuals suffer from severe curse of dimensionality and convergence stagnation. At 16×16 (148,480 weights and biases), both analytical backpropagation and evolutionary search converge efficiently.</li>
</ul>

<h2 class="section-title">4. Preprocessing & Feature Engineering</h2>
<ol>
    <li><strong>Pixel Intensity Normalization:</strong> Raw integer pixel values in [0, 255] are scaled to floating-point numbers in [0.0, 1.0] via <code>I_norm = I_raw / 255.0</code>. This guarantees numerical stability during matrix multiplications and prevents early activation saturation.</li>
    <li><strong>Grayscale Input Generation (ITU-R BT.601):</strong> To produce physiologically accurate luminance inputs from normalized RGB targets, we apply the standard ITU-R BT.601 perceptual formula:
    <code>Y = 0.299 &times; R + 0.587 &times; G + 0.114 &times; B</code>, accounting for human trichromatic sensitivity (greatest sensitivity to green, moderate to red, lowest to blue).</li>
    <li><strong>Matrix Representations:</strong> Input Matrix: <code>X &isin; &reals;<sup>2000 &times; 256</sup></code>; Target Matrix: <code>Y &isin; &reals;<sup>2000 &times; 768</sup></code>. Unflattened arrays are preserved for visual rendering.</li>
</ol>

<h2 class="section-title">5. Sanity Checks & Statistical Verification</h2>
<table class="data-table">
    <thead>
        <tr>
            <th>Channel / Feature</th>
            <th class="center-text">Mean</th>
            <th class="center-text">Std Deviation</th>
            <th class="center-text">Minimum</th>
            <th class="center-text">Maximum</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Grayscale Input (Y)</strong></td>
            <td class="center-text"><strong>0.4643</strong></td>
            <td class="center-text"><strong>0.2169</strong></td>
            <td class="center-text">0.0000</td>
            <td class="center-text">1.0000</td>
        </tr>
        <tr>
            <td><strong>Red Channel (R)</strong></td>
            <td class="center-text">0.4877</td>
            <td class="center-text">0.2274</td>
            <td class="center-text">0.0000</td>
            <td class="center-text">1.0000</td>
        </tr>
        <tr>
            <td><strong>Green Channel (G)</strong></td>
            <td class="center-text">0.4656</td>
            <td class="center-text">0.2204</td>
            <td class="center-text">0.0000</td>
            <td class="center-text">1.0000</td>
        </tr>
        <tr>
            <td><strong>Blue Channel (B)</strong></td>
            <td class="center-text">0.3959</td>
            <td class="center-text">0.2288</td>
            <td class="center-text">0.0000</td>
            <td class="center-text">1.0000</td>
        </tr>
    </tbody>
</table>

<div class="img-container">
    <img src="{img_pairs_b64}" alt="Sample Grayscale Inputs and Color Targets">
    <div class="img-caption">Figure 1: Task 1 Sanity Check — Sample Grayscale Inputs vs. Ground Truth Color Targets (16x16 pixels).</div>
</div>

<div class="img-container">
    <img src="{img_dist_b64}" alt="Pixel Value Distributions">
    <div class="img-caption">Figure 2: Task 1 Pixel Value Distributions — Grayscale luminance (left) and RGB channel histograms (right).</div>
</div>

<h2 class="section-title">6. Dataset Splitting & Mini-Batch Partitioning</h2>
<ul>
    <li><strong>Training Set (70%):</strong> 1,400 samples &rarr; <code>X_train: (1400, 256)</code>, <code>Y_train: (1400, 768)</code></li>
    <li><strong>Validation Set (20%):</strong> 400 samples &rarr; <code>X_val: (400, 256)</code>, <code>Y_val: (400, 768)</code></li>
    <li><strong>Testing Set (10%):</strong> 200 samples &rarr; <code>X_test: (200, 256)</code>, <code>Y_test: (200, 768)</code></li>
    <li><strong>Batching:</strong> The 1,400 training samples are partitioned into <strong>14 mini-batches of exactly 100 samples each</strong>:
        <code>X_train_batches &isin; &reals;<sup>14 &times; 100 &times; 256</sup></code> and <code>Y_train_batches &isin; &reals;<sup>14 &times; 100 &times; 768</sup></code>.
    </li>
</ul>

<div class="page-break"></div>

<h1 class="part-title">Part II: Task 2 — Neural Network Architecture (Built From Scratch)</h1>

<h2 class="section-title">7. Neural Network Architecture Design</h2>
<p>
In accordance with Task 2 specifications:
</p>
<ol>
    <li><strong>Implementation Paradigm:</strong> Pure object-oriented Python class (<code>NeuralNetwork</code>) using native NumPy array/matrix operations (<code>@</code> matrix operator). Pre-built layer or model classes (<code>torch.nn</code>, <code>keras.layers</code>, <code>sklearn</code>) are strictly prohibited and avoided.</li>
    <li><strong>Input Layer Size:</strong> <code>n_in = 256</code> features, corresponding to flattened 16×16 grayscale pixel intensities.</li>
    <li><strong>Output Layer Size:</strong> <code>n_out = 768</code> features, corresponding to flattened 16×16×3 RGB color channel values.</li>
    <li><strong>Hidden Layer Topology:</strong> A balanced two-hidden-layer MLP architecture [128, 128] was chosen to provide non-linear capacity for learning luminance-to-chrominance mapping while maintaining parameter tractability.</li>
</ol>

<h3 class="subsection-title">Layer Dimension & Parameter Breakdown:</h3>
<table class="data-table">
    <thead>
        <tr>
            <th>Layer</th>
            <th class="center-text">Input Dim</th>
            <th class="center-text">Output Dim</th>
            <th class="center-text">Weight Matrix (W)</th>
            <th class="center-text">Bias Vector (b)</th>
            <th class="center-text">Activation</th>
            <th class="center-text">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Hidden Layer 1</strong></td>
            <td class="center-text">256</td>
            <td class="center-text">128</td>
            <td class="center-text">(256, 128)</td>
            <td class="center-text">(1, 128)</td>
            <td class="center-text"><strong>ReLU</strong></td>
            <td class="center-text">32,896</td>
        </tr>
        <tr>
            <td><strong>Hidden Layer 2</strong></td>
            <td class="center-text">128</td>
            <td class="center-text">128</td>
            <td class="center-text">(128, 128)</td>
            <td class="center-text">(1, 128)</td>
            <td class="center-text"><strong>ReLU</strong></td>
            <td class="center-text">16,512</td>
        </tr>
        <tr>
            <td><strong>Output Layer</strong></td>
            <td class="center-text">128</td>
            <td class="center-text">768</td>
            <td class="center-text">(128, 768)</td>
            <td class="center-text">(1, 768)</td>
            <td class="center-text"><strong>Sigmoid</strong></td>
            <td class="center-text">99,072</td>
        </tr>
        <tr style="font-weight: 600; background: #EDF2F7;">
            <td>Total</td>
            <td class="center-text">—</td>
            <td class="center-text">—</td>
            <td class="center-text">—</td>
            <td class="center-text">—</td>
            <td class="center-text">—</td>
            <td class="center-text"><strong>148,480 parameters</strong></td>
        </tr>
    </tbody>
</table>

<h2 class="section-title">8. Activation Functions Implemented From Scratch</h2>
<p>
<strong>Hidden Layers (ReLU & Derivative):</strong> <code>f(z) = max(0, z)</code> and analytical derivative <code>f'(z) = 1 if z > 0 else 0</code>. Implemented manually using <code>np.maximum(0, z)</code> and <code>(z > 0).astype(float)</code>. ReLU prevents gradient attenuation across layers, computes with extreme efficiency, and prepares the model for Task 3 Backpropagation.
</p>
<p>
<strong>Output Layer (Sigmoid & Derivative):</strong> <code>&sigma;(z) = 1 / (1 + exp(-clip(z, -500, 500)))</code> with analytical derivative <code>&sigma;'(z) = &sigma;(z) &times; (1 - &sigma;(z))</code>. As mandated by Task 2 Requirement 4 (<em>"matched to your normalized color range"</em>), target color pixels reside strictly in [0.0, 1.0]. The Sigmoid function naturally squashes all output activations into (0, 1), guaranteeing valid normalized RGB values without artificial clamping.
</p>

<h2 class="section-title">9. Weight and Bias Initialization Strategy</h2>
<p>
Weights and biases are initialized from scratch without external initializers:
</p>
<ul>
    <li><strong>ReLU Hidden Layers:</strong> Initialized using He/Kaiming normal scaling: <code>W ~ N(0, sqrt(2 / n_in))</code>. This preserves activation variance across layers, avoiding premature vanishing or exploding gradients.</li>
    <li><strong>Sigmoid Output Layer:</strong> Initialized using Xavier/Glorot normal scaling: <code>W ~ N(0, sqrt(1 / n_in))</code>, tailored for logistic units.</li>
    <li><strong>Biases:</strong> Initialized with small Gaussian noise scaled by 0.01: <code>b = np.random.randn(1, n_out) * 0.01</code> to break initial symmetry across neurons.</li>
</ul>

<h2 class="section-title">10. Forward Pass Propagation & Caching Mechanism</h2>
<p>
The forward pass is evaluated using vectorized NumPy operations:
<code>Z1 = X @ W1 + b1, A1 = ReLU(Z1)</code> &rarr;
<code>Z2 = A1 @ W2 + b2, A2 = ReLU(Z2)</code> &rarr;
<code>Z3 = A2 @ W3 + b3, Y_hat = Sigmoid(Z3)</code>.
To prepare directly for <strong>Task 3 (Backpropagation)</strong>, all intermediate pre-activations (Z) and activations (A) are stored in lists directly on the network instance (<code>self.Z</code> and <code>self.A</code>) during execution.
</p>

<h2 class="section-title">11. Genetic Algorithm Chromosome Compatibility (Task 5 Foresight)</h2>
<p>
To satisfy Task 5 (<em>"represent each individual as a full set of weights"</em>), the clean modular structure of <code>NeuralNetwork</code> provides:
</p>
<ul>
    <li><strong>Continuous Chromosome Representation:</strong> Concatenates all 148,480 weights and biases into a continuous 1D chromosome for evolutionary operators (crossover and Gaussian mutation).</li>
    <li><strong>Matrix Deserialization:</strong> Allows mapping chromosome vectors directly back into <code>network.weights</code> and <code>network.biases</code> for validation batch fitness evaluation.</li>
</ul>

<h2 class="section-title">12. Verification & Experimental Sanity Results</h2>
<p>
The network was evaluated on batch 0 of the training set (<code>X_train_batches[0]</code>, 100 samples):
</p>
<ul>
    <li><strong>Input Shape:</strong> (100, 256) &rarr; <strong>Output Shape:</strong> (100, 768), perfectly matching (100, 16×16×3).</li>
    <li><strong>Dynamic Range:</strong> Minimum predicted pixel intensity = <strong>0.0478</strong>, Maximum = <strong>0.9200</strong> (strictly bounded in [0, 1]).</li>
    <li><strong>Initial Un-Trained MSE Loss:</strong> <strong>0.071109</strong>, matching theoretical expectations for random Sigmoid activations against naturally distributed normalized pixel values.</li>
</ul>

<div class="img-container">
    <img src="{img_task2_b64}" alt="Task 2 Initial Predictions">
    <div class="img-caption">Figure 3: Task 2 Verification — Grayscale Input (top) vs. Initial Untrained Network Prediction (middle) vs. Ground Truth Target (bottom) across 6 sample images.</div>
</div>

<div class="page-break"></div>

<h1 class="part-title">Part III: Task 3 — Backpropagation, Implemented From Scratch (60 Marks)</h1>

<h2 class="section-title">13. Mathematical Derivation of Loss & Analytical Backpropagation</h2>
<p>
In accordance with <strong>Task 3</strong> and <strong>Regulation 8</strong>, all loss computations, partial derivatives, chain-rule error propagation, and gradient descent weight updates are derived and executed using plain NumPy matrix arithmetic (<code>@</code>, <code>.T</code>, <code>np.sum</code>) with zero automatic differentiation or external optimizer libraries.
</p>

<h3 class="subsection-title">13.1 Mean Squared Error (MSE) & Output Error Gradient</h3>
<p>
For a mini-batch of <em>m = 100</em> samples and <em>D = 768</em> RGB features, the MSE objective is:
</p>
<p style="text-align: center; font-style: italic; background: #F7FAFC; padding: 5px; border-radius: 4px; border: 1px solid #E2E8F0; margin: 4px 0;">
L(Y&#770;, Y) = (1 / (m &times; D)) &sum;<sub>i=1</sub><sup>m</sup> &sum;<sub>k=1</sub><sup>D</sup> (Y&#770;<sub>ik</sub> &minus; Y<sub>ik</sub>)<sup>2</sup>
&nbsp;&nbsp;&rArr;&nbsp;&nbsp;
&part;L / &part;A<sup>[3]</sup> = (2 / (m &times; D)) &times; (A<sup>[3]</sup> &minus; Y)
</p>

<h3 class="subsection-title">13.2 Layer-by-Layer Chain Rule Propagation</h3>
<ol>
    <li><strong>Output Layer (l = 3, Sigmoid):</strong> <code>&delta;<sup>[3]</sup> = (&part;L / &part;A<sup>[3]</sup>) &odot; &sigma;'(Z<sup>[3]</sup>)</code>, where <code>&sigma;'(z) = &sigma;(z)(1 &minus; &sigma;(z))</code>.</li>
    <li><strong>Parameter Gradients (Layer l &isin; &#123;3, 2, 1&#125;):</strong> <code>&part;L / &part;W<sup>[l]</sup> = (A<sup>[l&minus;1]</sup>)<sup>T</sup> @ &delta;<sup>[l]</sup></code> and <code>&part;L / &part;b<sup>[l]</sup> = &sum;<sub>i=1</sub><sup>m</sup> &delta;<sup>[l]</sup><sub>i,:</sub></code>.</li>
    <li><strong>Hidden Layers (l &isin; &#123;2, 1&#125;, ReLU):</strong> Upstream activation gradient <code>&part;L / &part;A<sup>[l]</sup> = &delta;<sup>[l+1]</sup> @ (W<sup>[l+1]</sup>)<sup>T</sup></code>, followed by <code>&delta;<sup>[l]</sup> = (&part;L / &part;A<sup>[l]</sup>) &odot; ReLU'(Z<sup>[l]</sup>)</code> where <code>ReLU'(z) = 1 if z &gt; 0 else 0</code>.</li>
    <li><strong>Gradient Descent Update Rule:</strong> <code>W<sup>[l]</sup> &larr; W<sup>[l]</sup> &minus; &eta; (&part;L / &part;W<sup>[l]</sup>)</code> and <code>b<sup>[l]</sup> &larr; b<sup>[l]</sup> &minus; &eta; (&part;L / &part;b<sup>[l]</sup>)</code>.</li>
</ol>

<h2 class="section-title">14. Numerical Gradient Check & Learning Rate Tuning</h2>
<ul>
    <li><strong>Finite-Difference Verification (&epsilon; = 10<sup>&minus;6</sup>):</strong> Analytical gradients from <code>backward_pass()</code> matched central numerical derivatives <code>(L(w+&epsilon;) &minus; L(w&minus;&epsilon;)) / (2&epsilon;)</code> with a maximum relative error of <strong>1.64 &times; 10<sup>&minus;7</sup></strong> across all layers.</li>
    <li><strong>Learning Rate Tuning (&eta; = 0.1 &rarr; 15.0):</strong> Because element-wise MSE divides by <code>m &times; D = 76,800</code>, starting around <code>&eta; = 0.1</code> produces tiny steps (40-epoch Val MSE = 0.0559). Sweeping <code>&eta; &isin; &#123;0.1, 1.0, 5.0, 15.0, 30.0, 50.0&#125;</code> identified <strong><code>LEARNING_RATE = 15.0</code></strong> (40-epoch Val MSE = 0.0342; 300-epoch Val MSE = 0.0145) as optimal for smooth, non-oscillating descent.</li>
</ul>

<h2 class="section-title">15. Mini-Batch Training, Validation & Single Final Test Results</h2>
<p>
Training ran for <strong>300 epochs</strong> (14 mini-batches/epoch = 4,200 gradient updates, total runtime <strong>53.77s</strong>), evaluating validation once per epoch and running the held-out test pass <strong>strictly once</strong> after training completed:
</p>

<table class="data-table">
    <thead>
        <tr>
            <th class="center-text">Checkpoint</th>
            <th class="center-text">Training MSE</th>
            <th class="center-text">Validation MSE</th>
            <th>Observation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="center-text"><strong>Untrained (Epoch 0)</strong></td>
            <td class="center-text">0.071109 (Batch 0)</td>
            <td class="center-text"><strong>0.070611</strong></td>
            <td>Random He/Xavier initial baseline</td>
        </tr>
        <tr>
            <td class="center-text">Epoch 1</td>
            <td class="center-text">0.056027</td>
            <td class="center-text">0.055031</td>
            <td>Rapid luminance &amp; bias alignment</td>
        </tr>
        <tr>
            <td class="center-text">Epoch 50</td>
            <td class="center-text">0.027759</td>
            <td class="center-text">0.026243</td>
            <td>Smooth downward trajectory</td>
        </tr>
        <tr>
            <td class="center-text">Epoch 100</td>
            <td class="center-text">0.020476</td>
            <td class="center-text">0.020091</td>
            <td>&gt;71% MSE reduction from baseline</td>
        </tr>
        <tr>
            <td class="center-text">Epoch 200</td>
            <td class="center-text">0.016072</td>
            <td class="center-text">0.016346</td>
            <td>No overfitting (validation tracks training)</td>
        </tr>
        <tr style="font-weight: 600; background: #EDF2F7;">
            <td class="center-text">Epoch 300 (Final)</td>
            <td class="center-text"><strong>0.013692</strong></td>
            <td class="center-text"><strong>0.014522</strong></td>
            <td>Converged (79.4% Val MSE reduction)</td>
        </tr>
        <tr style="font-weight: 600; background: #E6FFFA;">
            <td class="center-text">Final Test Set (Once)</td>
            <td class="center-text">&mdash;</td>
            <td class="center-text"><strong>0.015382</strong></td>
            <td>Held-out Test MSE (200 unseen images)</td>
        </tr>
    </tbody>
</table>

<div class="img-container">
    <img src="{img_task3_curve_b64}" alt="Task 3 Convergence Curve">
    <div class="img-caption">Figure 4: Task 3 Training &amp; Validation MSE Convergence across 300 epochs (LR = 15.0) alongside Final Test MSE.</div>
</div>

<div class="img-container">
    <img src="{img_task3_preview_b64}" alt="Task 3 Test Set Predictions">
    <div class="img-caption">Figure 5: Visual sanity check on 6 unseen test images after 300 epochs of from-scratch backpropagation (Grayscale Input, Trained Output, True Color).</div>
</div>

<div class="page-break"></div>

<h1 class="part-title">Part IV: Task 4 — Evaluation and Results (25 Marks)</h1>

<h2 class="section-title">16. Learning Curves, Generalization &amp; Final Test Metrics</h2>
<p>
Training and validation MSE were recorded across all 300 epochs, and the held-out test set (200 unseen animal images) was evaluated <strong>strictly once</strong> after training converged:
</p>

<table class="data-table">
    <thead>
        <tr>
            <th>Evaluation Metric</th>
            <th class="center-text">Value</th>
            <th>Interpretation &amp; Context</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Final Test MSE</strong></td>
            <td class="center-text"><strong>0.015382</strong></td>
            <td><strong>78.2% reduction</strong> from untrained baseline (0.070611)</td>
        </tr>
        <tr>
            <td><strong>Final Test RMSE / MAE</strong></td>
            <td class="center-text"><strong>0.124022 / 0.094342</strong></td>
            <td>Average normalized pixel error on [0, 1]</td>
        </tr>
        <tr>
            <td><strong>Final Test PSNR</strong></td>
            <td class="center-text"><strong>18.13 dB</strong></td>
            <td>Peak Signal-to-Noise Ratio across 200 unseen test images</td>
        </tr>
        <tr>
            <td><strong>Per-Channel Test MSE (R / G / B)</strong></td>
            <td class="center-text"><strong>0.0148 / 0.0112 / 0.0201</strong></td>
            <td>Green is lowest (58.7% weight in BT.601 gray); Blue is highest (11.4%)</td>
        </tr>
        <tr>
            <td><strong>Generalization Gap (Val &minus; Train)</strong></td>
            <td class="center-text"><strong>~0.0008</strong></td>
            <td>Validation (0.014522) tightly tracks training (0.013692) &rarr; zero overfitting</td>
        </tr>
    </tbody>
</table>

<div class="img-container">
    <img src="{img_task4_curves_b64}" alt="Task 4 Learning Curves">
    <div class="img-caption">Figure 6: Task 4 Training &amp; Validation MSE vs. Epoch (left) and Generalization Gap across 300 epochs (right).</div>
</div>

<div class="img-container">
    <img src="{img_task4_colors_b64}" alt="Task 4 10 Test Colorizations">
    <div class="img-caption">Figure 7: Side-by-side comparison of 10 unseen test images (Grayscale Input, Network Colorized Prediction, True Color Ground Truth).</div>
</div>

<div class="page-break"></div>

<h2 class="section-title">17. Hyperparameter Sensitivity Study &amp; Observed Limitations</h2>
<p>
To isolate which design choices mattered most, we conducted a controlled 30-epoch validation sweep across Learning Rate, Batch Size, and Hidden Width (keeping <code>X_test</code> untouched):
</p>
<ul>
    <li><strong>Learning Rate (Most Critical):</strong> Because element-wise MSE divides gradients by <code>m &times; D = 76,800</code>, <code>LR=0.1</code> barely moved (30-epoch Val MSE = <code>0.056235</code>), whereas <code>LR=15.0</code> descended cleanly (<code>0.038633</code> at epoch 30; <code>0.014522</code> at epoch 300) and <code>LR=30.0</code> introduced step oscillations.</li>
    <li><strong>Epochs:</strong> Increasing from the 10-epoch minimum (Val MSE <code>0.043586</code>) to 300 epochs (Val MSE <code>0.014522</code>) was essential for sharp luminance &amp; chrominance alignment.</li>
    <li><strong>Batch Size &amp; Hidden Width:</strong> Smaller mini-batches (<code>B=50</code> &rarr; <code>0.027226</code> at 30 epochs) perform twice as many updates per epoch; wider hidden layers (<code>[256, 128]</code> &rarr; <code>0.032242</code>) provide extra spatial capacity at higher parameter cost.</li>
    <li><strong>MSE Desaturation &amp; Bottleneck Limitations:</strong> Plain MSE minimizes expected squared pixel error, averaging ambiguous colors toward muted earth/olive tones (visible on vivid pink/yellow objects in Figure 7). Additionally, a dense MLP bottleneck smooths fine edges compared to convolutional U-Nets or Lab-space chrominance prediction.</li>
</ul>

<div class="img-container">
    <img src="{img_task4_hp_b64}" alt="Task 4 Hyperparameter Study">
    <div class="img-caption">Figure 8: Controlled 30-epoch Validation MSE sensitivity across Learning Rate, Batch Size, and Hidden Layer Width.</div>
</div>

<div class="page-break"></div>

<h1 class="part-title">Part V: Task 5 — Genetic Algorithm Comparison (20 Marks)</h1>

<h2 class="section-title">18. Genetic Algorithm Implementation (From Scratch)</h2>
<p>
Preserving the exact <strong>Task 2 architecture</strong> <code>[256, 128, 128, 768]</code> (<strong>148,480</strong> parameters) and the <strong>same MSE error function</strong> as Task 3, we implemented a complete evolutionary optimizer using plain NumPy:
</p>
<ul>
    <li><strong>Chromosome Encoding:</strong> Each individual is a 1D <code>float32</code> vector of 148,480 genes representing all weights and biases, initialized with Task 2 He/Xavier scales.</li>
    <li><strong>Selection &amp; Crossover:</strong> Tournament Selection (<code>k = 3</code>) combined with Hybrid Crossover (50% uniform gene swap, 50% arithmetic blend <code>&alpha;p<sub>1</sub> + (1&minus;&alpha;)p<sub>2</sub></code>).</li>
    <li><strong>Mutation &amp; Replacement:</strong> Sparse Gaussian mutation (<code>mut_rate = 0.02</code>) with linearly annealing step size (<code>&sigma;: 0.02 &rarr; 0.005</code>) and elitist generational replacement (<code>ELITISM = 2</code>, <code>POP_SIZE = 20</code>, <code>100</code> generations = 2,000 full-train evaluations).</li>
</ul>

<h2 class="section-title">19. Empirical Comparison: Backpropagation vs. Genetic Algorithm</h2>
<table class="data-table">
    <thead>
        <tr>
            <th>Comparison Criterion</th>
            <th class="center-text">Backpropagation (Task 3)</th>
            <th class="center-text">Genetic Algorithm (Task 5)</th>
            <th>Key Finding</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Training Budget &amp; Runtime</strong></td>
            <td class="center-text">300 epochs (<strong>54.10 s</strong>)</td>
            <td class="center-text">100 gens &times; 20 pop (<strong>36.05 s</strong>)</td>
            <td>GA evaluates fast in <code>float32</code>, but stalls early</td>
        </tr>
        <tr>
            <td><strong>Convergence Speed</strong></td>
            <td class="center-text">Continuous drop &rarr; 0.0145</td>
            <td class="center-text">Plateaus after Gen ~10 at 0.0552</td>
            <td><strong>Backprop</strong> drops MSE continuously per second</td>
        </tr>
        <tr>
            <td><strong>Final Train / Val MSE</strong></td>
            <td class="center-text"><strong>0.013692 / 0.014522</strong></td>
            <td class="center-text">0.054435 / 0.055180</td>
            <td><strong>Backprop</strong> achieves ~4&times; lower error</td>
        </tr>
        <tr style="font-weight: 600; background: #E6FFFA;">
            <td><strong>Final Held-Out Test MSE</strong></td>
            <td class="center-text"><strong>0.015382</strong></td>
            <td class="center-text">0.055276</td>
            <td><strong>Backprop is 3.6&times; more accurate</strong> on unseen data</td>
        </tr>
        <tr>
            <td><strong>3-Seed Stability (40 iters)</strong></td>
            <td class="center-text">0.03189 &plusmn; 0.00204</td>
            <td class="center-text">0.05537 &plusmn; 0.00003</td>
            <td>GA has tiny seed variance because all runs stall at the mean</td>
        </tr>
    </tbody>
</table>

<p>
<strong>Why Backpropagation Wins in 148,480 Dimensions:</strong> Backpropagation uses exact first-order partial derivatives (<code>&nabla;<sub>W</sub>L</code>) to adjust all 148,480 parameters in coordinated descent toward natural animal colors. In contrast, a zeroth-order Genetic Algorithm relies on random mutations across 148,480 dimensions without directional guidance—after suppressing initial He/Xavier variance during Generations 1–10, it stagnates near the constant dataset-mean prediction (<code>~0.0552</code>, flat gray-brown outputs).
</p>

<div class="img-container">
    <img src="{img_task5_cmp_b64}" alt="Task 5 GA vs Backprop Comparison">
    <div class="img-caption">Figure 9: Quantitative Comparison — Iteration Trajectory (left), Wall-Clock Convergence Speed (center), and Final Error &amp; 3-Seed Stability (right).</div>
</div>

<div class="page-break"></div>

<div class="img-container">
    <img src="{img_task5_vis_b64}" alt="Task 5 Visual Comparison">
    <div class="img-caption">Figure 10: Visual Comparison on 6 Unseen Test Images — Grayscale Input, Backprop Colorization, Genetic Algorithm Output, and True Color Ground Truth.</div>
</div>

<h2 class="section-title">20. Final Summary Compliance Matrix (All 150 Marks &mdash; Tasks 1&ndash;5)</h2>
<table class="data-table">
    <thead>
        <tr>
            <th style="width: 22%;">Requirement</th>
            <th style="width: 48%;">Course Specification</th>
            <th class="center-text" style="width: 14%;">Status</th>
            <th style="width: 16%;">Evidence</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Regulations 1&ndash;5</strong></td>
            <td>Team of 5, full cover details, report, matching <code>.zip</code> &amp; <code>.ipynb</code> filenames</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Verified</td>
        </tr>
        <tr>
            <td><strong>Regulations 6&ndash;8</strong></td>
            <td>Single cell per task starting with <code>#Task</code> comments; 100% pure NumPy from scratch</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Cells 2, 5, 8, 11, 14</td>
        </tr>
        <tr>
            <td><strong>Task 1 (10 Marks)</strong></td>
            <td>CIFAR-10 Animals (16&times;16), [0,1] norm, BT.601 gray, 70/20/10 split, 14 batches of 100</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Cell 2 &amp; Figs 1&ndash;2</td>
        </tr>
        <tr>
            <td><strong>Task 2 (15 Marks)</strong></td>
            <td><code>NeuralNetwork</code> [256, 128, 128, 768] (148,480 params), ReLU &amp; Sigmoid, He/Xavier init</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Cell 5 &amp; Fig 3</td>
        </tr>
        <tr>
            <td><strong>Task 3 (60 Marks)</strong></td>
            <td>Forward pass, MSE loss, chain-rule backprop (<code>1.64e-07</code> check), tuned SGD, 300 epochs, single test pass</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Test MSE: 0.015382</td>
        </tr>
        <tr>
            <td><strong>Task 4 (25 Marks)</strong></td>
            <td>Loss curves, final test metrics, 10 side-by-side images, hyperparameter study &amp; limitations</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>PSNR: 18.13 dB</td>
        </tr>
        <tr>
            <td><strong>Task 5 (20 Marks)</strong></td>
            <td>From-scratch GA (148,480 chromosomes, selection/crossover/mutation/elitism) &amp; 3-axis comparison</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>GA MSE: 0.055276</td>
        </tr>
    </tbody>
</table>

</body>
</html>
"""

weasyprint.HTML(string=html_content).write_pdf("Project_Report.pdf")
print("Compiled Project_Report.pdf successfully!")
