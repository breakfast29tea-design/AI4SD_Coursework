# COMP0173 AI for Sustainable Development Coursework 2
## Introduction:
This repository contains code for two cases:
  1. Clone of original Amazon deforestation segmentation (Original_clone)
  2. Glacier calving front detection (Glacier_project)

**More details & Technical Implementation Report (part A in coursework 2) is documented in `Part_A_Tech.md`**

Note: Datasets are not included due to size constraints. Please download raw datasets and replace them with the empty folders `data`.
* For Clone Case:
    - original paper: John, D., & Zhang, C. (2022). An attention-based U-Net for detecting deforestation within satellite sensor imagery. *International Journal of Applied Earth Observations and Geoinformation*, 107, 102685.
    (https://www.sciencedirect.com/science/article/pii/S0303243422000113)
    - repo: https://github.com/davej23/attention-mechanism-unet
    - AMAZON dataset: https://zenodo.org/records/4498086#.YMh3GfKSmCU
* For Glacier Project:
    - CaFFe dataset: https://doi.pangaea.de/10.1594/PANGAEA.940950?

## Repository Structure:
AI4SD_Coursework  
  * Original_clone
    - clone_preprocess.py
    - clone_model.py  
  * Glacier_project   
    - Glacier_preprocess.py  
    - Glacier_model&evaluation.py  
  * environment.yml
  * README.md
  * data

## Instructions:
1. Clone this repository.
2. Download the raw datasets:
   - `AMAZON` for the clone case.  
   - `S2GLC` for the glacier case.  
3. Replace the existing `data` folder with the downloaded dataset folders.  
   (Please do NOT put the dataset inside the `data` folder — replace the folder itself.)
4. Run the preprocessing scripts. These will generate the processed data files.
5. Train the model by running the model scripts.
6. Evaluation using the test set (F1-score and IoU) is included inside the model scripts.

## Glacier_project:
The Glacier Project adapts the original Amazon Attention U-Net:
* Reduced layers (4 → 3)
* Simplified attention gates
* Input normalization changed from min-max per image to global scaling
* Mask definition: multi-class masks (`zones` in CaFFe) were redefined into binary segmentation as the target. 
