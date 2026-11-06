import os
import base64
import weasyprint

# Read images and encode to base64 for embedding
def get_b64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

img_pairs_b64 = get_b64_image("task1_sanity_check_pairs.png")
img_dist_b64 = get_b64_image("task1_pixel_distributions.png")

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CINN Course Project Report — Task 1</title>
<style>
    @page {{
        size: A4;
        margin: 15mm 14mm 15mm 14mm;
        @bottom-right {{
            content: "Page " counter(page) " of " counter(pages);
            font-size: 8.5pt;
            color: #718096;
        }}
        @bottom-left {{
            content: "DMET902 / CSEN1121 — Course Project Report (Winter 2026)";
            font-size: 8.5pt;
            color: #718096;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 9.8pt;
        line-height: 1.42;
        color: #2D3748;
        margin: 0;
    }}
    .header {{
        text-align: center;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 15px;
        margin-bottom: 18px;
    }}
    .header h1 {{
        font-size: 17pt;
        margin: 0 0 4px 0;
        color: #1A202C;
    }}
    .header h2 {{
        font-size: 12.5pt;
        font-weight: 600;
        margin: 0 0 6px 0;
        color: #4A5568;
    }}
    .header p {{
        font-size: 9.5pt;
        margin: 2px 0;
        color: #718096;
    }}
    .meta-box {{
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 18px;
    }}
    .meta-box table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .meta-box td {{
        padding: 4px 8px;
        font-size: 9.5pt;
    }}
    .meta-box td.label {{
        font-weight: 600;
        color: #4A5568;
        width: 25%;
    }}
    h2.section-title {{
        font-size: 12.5pt;
        color: #2B6CB0;
        border-bottom: 1.5px solid #E2E8F0;
        padding-bottom: 4px;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }}
    h3.subsection-title {{
        font-size: 11pt;
        color: #2D3748;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 8.8pt;
        page-break-inside: avoid;
    }}
    table.data-table th, table.data-table td {{
        border: 1px solid #CBD5E0;
        padding: 6px 10px;
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
        padding: 2px 6px;
        font-size: 8pt;
        font-weight: 600;
        background: #C6F6D5;
        color: #22543D;
        border-radius: 4px;
    }}
    .img-container {{
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
    }}
    .img-container img {{
        max-width: 95%;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
    }}
    .img-caption {{
        font-size: 8.5pt;
        color: #718096;
        margin-top: 4px;
        font-style: italic;
    }}
    code {{
        font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 8.5pt;
        background: #EDF2F7;
        padding: 2px 5px;
        border-radius: 3px;
        color: #C53030;
    }}
    pre {{
        background: #2D3748;
        color: #F7FAFC;
        padding: 10px;
        border-radius: 5px;
        font-size: 8.5pt;
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
    <p>Instructors: Prof. Dr. Mohammed Abdel-Megeed Salem, Eng. Yomna Islam Youssef Alayary</p>
</div>

<div class="meta-box">
    <table>
        <tr>
            <td class="label">Project Title:</td>
            <td><strong>Image Colorization Using From-Scratch Neural Networks & Genetic Algorithm</strong></td>
        </tr>
        <tr>
            <td class="label">Milestone / Scope:</td>
            <td><strong>Task 1 — Dataset, Preprocessing, and Splitting (10 Marks)</strong></td>
        </tr>
        <tr>
            <td class="label">Submission Deadline:</td>
            <td>Friday, November 6, 2026</td>
        </tr>
        <tr>
            <td class="label">Team Name:</td>
            <td><code>404 Intelligence Not Found</code> (Archive: <code>CINN_WS2026_Project_404_Intelligence_Not_Found</code>)</td>
        </tr>
    </table>
</div>

<h3 class="subsection-title" style="margin-top: 15px;">Team Members</h3>
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

<h2 class="section-title">1. Executive Summary & Objective</h2>
<p>
The objective of the CINN course project is to build an end-to-end image colorization engine implemented entirely from scratch without high-level deep learning frameworks (e.g. PyTorch, Keras, TensorFlow). The system learns to map single-channel grayscale images to plausible 3-channel RGB color representations, first through analytical backpropagation (Task 3) and subsequently through evolutionary weight optimization via a Genetic Algorithm (Task 5).
</p>
<p>
Task 1 establishes the mathematical and operational data pipeline required for all downstream tasks. It guarantees that images are cleanly preprocessed, appropriately sized, verified through statistical and visual sanity checks, and cleanly partitioned into training, validation, and testing mini-batches.
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
<p style="text-align: center; font-style: italic; background: #F7FAFC; padding: 6px; border-radius: 4px; border: 1px solid #E2E8F0;">
Downsampled(i, j, c) = &frac14; &sum;<sub>u=0</sub><sup>1</sup> &sum;<sub>v=0</sub><sup>1</sup> Original(2i + u, 2j + v, c)
</p>
<ul>
    <li><strong>Feature Dimensionality at 32×32:</strong> Input = 1,024 grayscale features; Output = 3,072 RGB features. A single hidden layer of 128 neurons requires (1,024 × 128) + (128 × 3,072) &approx; <strong>524,288 weights</strong>.</li>
    <li><strong>Feature Dimensionality at 16×16:</strong> Input = 256 grayscale features; Output = 768 RGB features. A hidden layer of 64 neurons requires (256 × 64) + (64 × 768) &approx; <strong>65,536 weights</strong> — an <strong>8&times; reduction in computational complexity</strong>.</li>
    <li><strong>Critical for Task 5 (Genetic Algorithm):</strong> Chromosomes representing 500k+ real-valued parameters across a population of 50 individuals suffer from severe curse of dimensionality and convergence stagnation. At 16×16 (~65k weights), the genetic algorithm evaluates fitness and converges efficiently.</li>
</ul>

<div class="page-break"></div>

<h2 class="section-title">4. Preprocessing & Feature Engineering</h2>
<ol>
    <li><strong>Pixel Intensity Normalization:</strong> Raw integer pixel values in [0, 255] are scaled to floating-point numbers in [0.0, 1.0] via <code>I_norm = I_raw / 255.0</code>. This guarantees numerical stability during matrix multiplications and prevents early activation saturation.</li>
    <li><strong>Grayscale Input Generation (ITU-R BT.601):</strong> To produce physiologically accurate luminance inputs from normalized RGB targets, we apply the standard ITU-R BT.601 perceptual formula:<br>
    <div style="text-align: center; margin: 6px 0; font-weight: 600;">Y = 0.299 &times; R + 0.587 &times; G + 0.114 &times; B</div>
    accounting for human trichromatic sensitivity (greatest sensitivity to green, moderate to red, lowest to blue).</li>
    <li><strong>Matrix Representations:</strong>
        <ul>
            <li>Input Matrix (X): <code>X &isin; &reals;<sup>2000 &times; 256</sup></code>, grayscale intensities in [0.0, 1.0].</li>
            <li>Target Matrix (Y): <code>Y &isin; &reals;<sup>2000 &times; 768</sup></code>, RGB color values in [0.0, 1.0].</li>
            <li>Unflattened arrays (2000, 16, 16) and (2000, 16, 16, 3) are preserved for visual rendering in Tasks 4 and 5.</li>
        </ul>
    </li>
</ol>

<h2 class="section-title">5. Sanity Checks & Statistical Verification</h2>
<p>
Automated assertions strictly enforce valid value ranges:
<code>assert np.all(X_flat &gt;= 0.0) and np.all(X_flat &lt;= 1.0)</code> and <code>assert np.all(Y_flat &gt;= 0.0) and np.all(Y_flat &lt;= 1.0)</code>.
</p>

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
    <div class="img-caption">Figure 1: Sanity Check — Sample Grayscale Inputs vs. Ground Truth Color Targets (16x16 pixels).</div>
</div>

<div class="img-container">
    <img src="{img_dist_b64}" alt="Pixel Value Distributions">
    <div class="img-caption">Figure 2: Pixel Value Distributions — Grayscale luminance (left) and RGB channel histograms (right).</div>
</div>

<h2 class="section-title">6. Dataset Splitting & Mini-Batch Partitioning</h2>
<p>
In accordance with Task 1 Requirements 5 and 6:
</p>
<ul>
    <li><strong>Training Set (70%):</strong> 1,400 samples &rarr; <code>X_train: (1400, 256)</code>, <code>Y_train: (1400, 768)</code></li>
    <li><strong>Validation Set (20%):</strong> 400 samples &rarr; <code>X_val: (400, 256)</code>, <code>Y_val: (400, 768)</code></li>
    <li><strong>Testing Set (10%):</strong> 200 samples &rarr; <code>X_test: (200, 256)</code>, <code>Y_test: (200, 768)</code></li>
    <li><strong>Batching:</strong> The 1,400 training samples are partitioned into <strong>14 mini-batches of exactly 100 samples each</strong>:
        <code>X_train_batches &isin; &reals;<sup>14 &times; 100 &times; 256</sup></code> and <code>Y_train_batches &isin; &reals;<sup>14 &times; 100 &times; 768</sup></code>.
    </li>
</ul>

<h2 class="section-title">7. Notes to be Considered in Grading (Regulation 3)</h2>
<ol>
    <li><strong>Single Cell Execution (Regulation 6):</strong> The entire data processing pipeline is consolidated into ONE self-contained cell block with the required comment tags: <code>#Task 1</code>, <code>#Task requirement</code>, and <code>#Task overview</code>.</li>
    <li><strong>Zero Pre-Built High-Level Layers (Regulation 8):</strong> All operations (downsampling, normalization, batch partitioning) use plain NumPy operations without external deep learning libraries.</li>
    <li><strong>Robust Dataset Discovery:</strong> Automatically locates CIFAR-10 whether mounted uncompressed in Kaggle inputs, uploaded as a raw archive (e.g. <code>cifar-10-python.tar.gz</code> or unextended <code>target_archive</code>), or stored in a local directory.</li>
    <li><strong>Reproducibility:</strong> Seed is fixed with <code>np.random.seed(42)</code>, guaranteeing identical splits and figures on any machine.</li>
</ol>

<h2 class="section-title">8. Summary Compliance Matrix</h2>
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
            <td>Ready</td>
        </tr>
        <tr>
            <td><strong>Regulation 3 & 4</strong></td>
            <td>Report with operations, results, team details & grading notes</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Cover & Sections 1-7</td>
        </tr>
        <tr>
            <td><strong>Regulation 5</strong></td>
            <td>Jupyter Notebook named identically to zip file</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td><code>.ipynb</code> verified</td>
        </tr>
        <tr>
            <td><strong>Regulation 6</strong></td>
            <td>Task in ONE single cell starting with #Task comments</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Cell 1</td>
        </tr>
        <tr>
            <td><strong>Regulation 7</strong></td>
            <td>Commented code explaining steps</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Clean comments</td>
        </tr>
        <tr>
            <td><strong>Regulation 8</strong></td>
            <td>From-scratch NumPy implementation, no plagiarism</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>Pure NumPy</td>
        </tr>
        <tr>
            <td><strong>Task 1.1 - 1.6</strong></td>
            <td>Dataset, Grayscale/RGB, Normalization, Sanity plots, 70/20/10 split, Batches of 100</td>
            <td class="center-text"><span class="badge">Complete</span></td>
            <td>All 6 steps passed</td>
        </tr>
    </tbody>
</table>

</body>
</html>
"""

# Compile to PDF using WeasyPrint
weasyprint.HTML(string=html_content).write_pdf("Project_Report.pdf")
print("Compiled Project_Report.pdf successfully!")
