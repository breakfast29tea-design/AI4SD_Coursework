# COMP0173 AI for Sustainable Development Coursework 2
## Introduction:
This repository contains code for two cases:
  1. Clone of original Amazon deforestation segmentation (Original_clone)
  2. Glacier calving front detection (Glacier_project)

Note: Datasets are not included due to size constraints. Please download raw datasets and replace them with the empty folders `data`.
* For clone case:
    - original paper: John, D., & Zhang, C. (2022). An attention-based U-Net for detecting deforestation within satellite sensor imagery. *International Journal of Applied Earth Observations and Geoinformation*, 107, 102685.
    (https://www.sciencedirect.com/science/article/pii/S0303243422000113)
    - repo: https://github.com/davej23/attention-mechanism-unet
    - AMAZON dataset: https://zenodo.org/records/4498086#.YMh3GfKSmCU
* For Glacier project:
    - CaFFe dataset: https://doi.pangaea.de/10.1594/PANGAEA.940950?

## Repository Structure:
AI4SD_Coursework  
  * Clone
    - preprocess
    - model  
  * Glacier_project   
    - preprocess  
    - model  
  * environment.yml
  * README.md

## Instructions:
1. Clone this repo.
2. Download the raw dataset and replace them with the `data` in both `Original_clone` & `Glacier_project`.
3. Run the preprocess script, this would create a new data file. Please ensure this is horizontal with the two project files.
4. Train the model: run the model script.
5. Evaluation using test set (F1 & IoU) is included inside the model script.

## Glacier_project:
The Glacier Project adapts the original Amazon Attention U-Net:
* Reduced layers (4 → 3)
* Simplified attention gates
* Input normalization changed from min-max per image to global scaling
* Mask definition: multi-class masks (`zones` in CaFFe) were redefined into binary segmentation as the target. 
