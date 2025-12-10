# Part A – Technical Implementation

## 1. Replicating Baseline Method
### 1.1 Cloning the original repository
* The original implementation was obtained from the following public repository: https://github.com/davej23/attention-mechanism-unet
* The repository was successfully cloned and used as the baseline for this project.

### 1.2 Environment setup
* All dependencies and environments were managed using Conda.  
* The configuration file is provided in `environment.yml`.
* Due to compatibility issues across library versions, the code could only be executed reliably under specific versions, including **Python 3.8**. These constraints were carefully documented to ensure reproducibility.

### 1.3 Reproduction of baseline results within ±5% of original paper metrics
* The original study provides three datasets (3-band RGB, 4-band AMAZON, and 4-band Atlantic Forest).  In this project, the **4-band AMAZON** dataset was selected for baseline replication.
* The reproduced results were compared against the original reported metrics:
  
  |Item|F1-score|IoU|
  |---|---|---|
  |Original paper|0.9581|0.9199|
  |Clone result|0.9626|0.9278|

### 1.4 Reproducible scripts and notebooks
Please find files in `Original_clone`. 

---
## 2. New Context & SDG Motivation

Here is the brief summary of the project's goal:

|Item|New Context|
|---|---|
|Relevant Challenge|Glacier Calving Detection|
|Tranferability of AI Methodology|Attention U-Net|
|Region of Choice|Polar Circles: Greenland, Antarctic, Alaska|

### 2.1 Problem Definition
Glaciers are primarily located in polar and high-latitude regions, far away from most global population. However, the retreat of glacier calving fronts has direct and indirect impacts on global sea-level rise and climate systems, for example, the abnormal weather pattern.  
* Definition: glacier calving refers to the process where large chunks of ice break off from the terminus of a glacier into the ocean. This phenomenon is a major contributor to glacier mass loss.
* Mechanism works as a feedback loop: ocean warming -> ice melt -> calving front retreat -> lower resistance -> faster flow rate -> more ice loss at front -> further retreat of calving front
* All processes in the loop cause sea-level rising.  
 
Reference:  
- https://www.worldwildlife.org/resources/explainers/why-are-glaciers-and-sea-ice-melting/  
- https://www.climate.gov/news-features/understanding-climate/climate-change-mountain-glaciers


### 2.2 SDG Alignment
This project aligns with **SDG 13 Climate Action** by developing automated methods for monitoring glacier calving front retreat. However, it is important to recognize that *technical monitoring alone does not directly mitigate climate change*. Therefore, its primary value lies in evidence generation, which can inform climate adaptation strategies, early-warning systems, and policy decision-making.  

### 2.3 Limitations and Ethics
* Limitations:
  - SAR-based glacier imagery relies on remote sensing data, which may contain inherent noise and geometric distortions.
  - The adopted dataset combines imagery from multiple satellite missions, which may involve cross-national governance and access constraints.
  - A critical limitation is that **improvements in detection accuracy could NOT automatically translate into political or societal action, highlighting the gap between scientific capability and governance implementation**.

* Ethical Issues:  
  - Risk of misinterpretation: Inaccurate or overconfident model outputs could mislead public discourse or policy decisions.
  - Data sovereignty: Satellite data often originate from national or military-grade systems, raising concerns about data governance and access equity.
  - Accountability: AI systems cannot replace human responsibility in climate policy. They should be considered as decision-support tools rather than decision-makers.

### 2.4 Scalability and Sustainability
From a technical perspective, deep learning-based segmentation is scalable across large spatial areas and long temporal sequences. However, scalability is constrained by:
* High computational and energy costs of model training.
* Dependency on continuous satellite data availability.
* Reliance on manually defined masks and human-driven annotation rules, limiting the speed, consistency, and scalability of ground truth generation.

From a sustainable perspective, while model simplification reduces computational cost, the comparison in this project showed that efficiency gains often trade off with performance, indicating an inherent tension between **sustainability of computation and accuracy of climate monitoring**.

### 2.5 Actionable Policy Recommendations
Based on the analysis above, several recommendations related to policy-making could be proposed:
* Investment in open-access climate satellite data to reduce global inequality in environmental monitoring.
* Use of AI-derived glacier retreat indicators as part of national climate risk assessment frameworks.
* Integration of automated glacier monitoring into early-warning system, especially for coastal regions.

---
## 3. Alternative Dataset
### 3.1 Dataset selection
CaFFe (https://doi.pangaea.de/10.1594/PANGAEA.940950) was chosen in this project.  
* It includes seven glaciers from 1995 to 2020:
  - Greenland (Jakobshavn Isbrae Glacier)
  - Alaska (Columbia Glacier)
  - Antarctica (Crane, Dinsmoore-Bombardier-Edgeworth, Mapple, Jorum and the Sjörgen-Inlet Glacier)
* To prevent leakage, there are five glaciers in training set (599 images, no validation set) while two in the test set (122 images). 
* The images have different spatial resolutions due to acquisition by multiple satellite platforms, including: Sentinel-1, TerraSAR-X, TanDEM-X, ENVISAT, European Remote Sensing Satellite 1&2, ALOS PALSAR, and RADARSAT-1.
* The dataset used in this project includes Synthetic Aperture Radar (SAR) and zones images.
* The zone masks categorize each pixel into four semantic classes:
  - Glacier
  - Rock outcrop
  - Ocean (including ice-melange) 
  - No information available (SAR shadows, layover regions, and areas outside the swath)  

### 3.2 Data access and ethics
* The dataset is openly available through the PANGAEA data repository.  
* Ethical considerations:
  - Responsible use of environmental data
  - Careful interpretation of climate-related results

### 3.3 Preprocessing pipeline
Please find `Glacier_preprocess.py` in `Glacier_project`. The script performs:
* Image–mask alignment  
* Resizing (to 512x512)
* Normalization: per-image contrast stretching (1–99 percentile) & global scaling
* Mild gamma correction (to adjust image brightness)
* Label generation for glacier front detection: multi-class zone masks are remapped to a binary segmentation task 
  Since the goal is to identify the calving front (retreating glacier vs ocean), the binary mask classifies pixels as either **ocean (background) or non-ocean (glacier foreground)**.

---
## 4. Model Adaptation
### 4.1 Architectural changes
The original Attention U-Net was adapted to the glacier calving front detection task for differences in **data modality (optical → SAR), input dimensions, and computational constraints**.  
Key modifications include:  
* **Input channels:** Original model used 4-band optical imagery; while the adapted model takes **single-channel SAR images**.  
* **Attention gate (AG) simplification:** The original AG was simplified to reduce computation while maintaining skip-connection modulation.
  - Replaced *Conv 1×1 + MaxPooling (2×2)* with a single **Conv 1×1 with stride = 2** (no separate max pooling).
  - Removed explicit channel repetition, for Keras/TensorFlow handles channel broadcasting automatically. 
* **Upsampling strategy:** Replaced *transpose convolution* with **simple upsampling + Conv2D** to reduce memory usage and computation time.  
* **Reduced depth:** Three-layer U-Net architecture (4 -> 3) was used to lower training complexity.

### 4.2 Hyperparameter tuning

|Hyperparameter|Original Model|Adapted Model|
|:---:|:---:|:---:|
|dropout|0.25|0.3|
|learning_rate|5e-4|1e-6|
|loss|binary cross entropy|dice loss|
|batch size|2|8|
|training epochs|60|30|

---
## 5. Evaluation
### 5.1 Performance comparison
The performance of all models are listed in the table:

|Model|F1-score|IoU|
|---|---|---|
|Original Deforestation|0.9581|0.9199|
|Clone Deforestation|0.9601|0.9233|
|Glacier Original Model|0.9367|0.8824|
|Adapted Glacier Model (Simplified)|0.9341|0.8778|

### 5.2 Metrics
F1 & Jaccard score (Intersection over Union, IoU) were suitable for evaluation, because:
- **F1-score** balances precision and recall for class-imbalanced segmentation.  
- **IoU** quantifies the overlap between predicted and ground-truth masks, providing a robust measure of spatial agreement.

### 5.3 Statistical analysis
A direct statistical comparison between the original baseline model and the adapted model was not conducted, for models were trained on fundamentally different data modalities: the former was trained on 4-band optical imagery (RGB + NIR), whereas the latter was trained on single-channel SAR imagery. Due to the **mismatch in input distributions** and learning conditions, a paired statistical test under identical experimental settings was not methodologically appropriate.
As a result, a paired t-test was conducted to compare the performance of the **original glacier model** and the **simplified glacier model**, as both networks were trained and evaluated on the same SAR dataset.

The result *t=-7.5, p<0.001* indicates a statistically significant performance difference between the two models, with the original glacier model outperforming the simplified version, which is consistent with the quantitative performance trends observed in Section 5.1.

### 5.4 Failure case analysis
1. **Mask Definition Challenge**  
Foreground vs. Background (Multi-class → Binary Segmentation)
The original dataset contained multi-class zone labels. However, for calving front detection, only the glacier–ocean boundary is relevant. Several iterations were required to redefine the mask into a binary format (ocean vs. non-ocean). Early experiments showed decreasing F1-score during training, indicating that ambiguous class definitions can prevent effective learning. Refining the mask definition significantly improved training stability.

2. **Data Leakage Risk: Train–Validation Split Debugging**  
Unexpectedly high validation performance (near-perfect validation metrics) revealed a risk of data leakage. This was traced to similarities in file naming patterns and spatial overlap between training and validation samples. Since satellite images from the same glacier region share highly similar spatial features, random file-based splitting can cause the model to recognise nearly identical scenes during both training and validation. This required careful re-splitting to ensure true spatial separation between training and validation data.

3. **Domain Shift Challenge: Optical vs. SAR Imagery**  
A major challenge in this project was the domain shift between the original optical 4-band (RGB + NIR) dataset and the adapted 1-band SAR dataset. Optical imagery captures surface reflectance, while SAR imagery is based on microwave backscatter, leading to very different texture patterns and noise characteristics. To mitigate this, contrast stretching and gamma correction were applied, though the intrinsic modality gap remains a fundamental limitation.
