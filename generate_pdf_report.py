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

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CINN Course Project Report — Tasks 1 & 2</title>
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
            <td><strong>Task 1 (Dataset & Splitting) & Task 2 (Neural Network Architecture)</strong></td>
        </tr>
        <tr>
            <td class="label">Task 2 Deadline:</td>
            <td>Sunday, November 15, 2026</td>
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

<h2 class="section-title">13. Summary Compliance Matrix (Tasks 1 & 2)</h2>
<table class="data-table">
    <thead>
        <tr>
            <th style="width: 20%;">Requirement</th>
            <th style="width: 50%;">Course Specification</th>
            <th class="center-text" style="width: 15%;">Status</th>
            <th style="width: 15%;">Evidence</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Regulation 2</strong></td>
            <td>Zip file named <code>CINN_WS2026_Project_[Teamname].zip</code></td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Submission Zip ready</td>
        </tr>
        <tr>
            <td><strong>Regulation 3 & 4</strong></td>
            <td>Report with operations, results, team details & grading notes</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Sections 1–13 detailed</td>
        </tr>
        <tr>
            <td><strong>Regulation 5</strong></td>
            <td>Jupyter Notebook named identically to zip file</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td><code>.ipynb</code> verified</td>
        </tr>
        <tr>
            <td><strong>Regulation 6</strong></td>
            <td>Every task in ONE single cell starting with #Task comments</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Cell 1 (Task 1) & Cell 2 (Task 2)</td>
        </tr>
        <tr>
            <td><strong>Regulation 7</strong></td>
            <td>Commented code explaining steps</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Clean, student-style comments</td>
        </tr>
        <tr>
            <td><strong>Regulation 8</strong></td>
            <td>From-scratch NumPy implementation, no pre-built layers</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Pure NumPy</td>
        </tr>
        <tr>
            <td><strong>Task 1.1 – 1.6</strong></td>
            <td>Dataset, Grayscale/RGB, Normalization, Sanity plots, 70/20/10 split, Batches of 100</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>All 6 steps passed</td>
        </tr>
        <tr>
            <td><strong>Task 2.1</strong></td>
            <td>Network class built from scratch (no imported layer/model)</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td><code>NeuralNetwork</code> class</td>
        </tr>
        <tr>
            <td><strong>Task 2.2</strong></td>
            <td>Input layer size = number of grayscale features</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td><code>n_in = 256</code></td>
        </tr>
        <tr>
            <td><strong>Task 2.3</strong></td>
            <td>Output layer size = number of color values</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td><code>n_out = 768</code></td>
        </tr>
        <tr>
            <td><strong>Task 2.4</strong></td>
            <td>Hidden layers/neurons chosen, manual ReLU & Sigmoid</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>[128, 128] with ReLU, Sigmoid & derivatives</td>
        </tr>
        <tr>
            <td><strong>Task 2.5</strong></td>
            <td>Initialize weights & biases with small random values from scratch</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>He/Xavier normal scaling & Gaussian noise biases</td>
        </tr>
    </tbody>
</table>

</body>
</html>
"""

weasyprint.HTML(string=html_content).write_pdf("Project_Report.pdf")
print("Compiled Project_Report.pdf successfully!")
