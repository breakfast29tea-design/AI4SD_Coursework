import os
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 

BASE_S2GLC = Path("./S2GLC")

RAW_ZONE_TRAIN_DIR = BASE_S2GLC / "zones" / "train" 
RAW_SAR_TRAIN_DIR = BASE_S2GLC / "sar_images" / "train" 

RAW_ZONE_TEST_DIR = BASE_S2GLC / "zones" / "test" 
RAW_SAR_TEST_DIR = BASE_S2GLC / "sar_images" / "test"

PROCESSED_DATA_ROOT = BASE_S2GLC.parent / "S2GLC_Processed_Full" 
PROCESSED_DATA_ROOT.mkdir(exist_ok=True)

IMAGE_SIZE = 512 
VALIDATION_SIZE = 0.20
RANDOM_SEED = 100

def resize_and_save_full_image(large_sar, large_mask, base_name: str, output_subdir: str):
    
    # target file：S2GLC_Processed_Full_Images_512_RAM/{train/val/test}/{images/masks}
    output_dir_images = PROCESSED_DATA_ROOT / output_subdir / "images"
    output_dir_masks = PROCESSED_DATA_ROOT / output_subdir / "masks"
    output_dir_images.mkdir(parents=True, exist_ok=True)
    output_dir_masks.mkdir(parents=True, exist_ok=True)
    
    # image resize to 512*512
    sar_resized = cv2.resize(large_sar, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # mask resize
    mask_resized = cv2.resize(large_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # global normalization
    normalized_sar = sar_resized.astype(np.float32) / 255.0
    
    # per-image contrast stretching (1~99 percentile)
    p1, p99 = np.percentile(sar, (1, 99))
    sar = np.clip((sar - p1) / (p99 - p1 + 1e-6), 0, 1)

    # mild gamma correction
    gamma = 1.2
    sar = np.power(sar, gamma)
    
    # binary threshold setting: target area (calving front: glacier + others vs sea)
    processed_mask = (mask_resized <= 127).astype(np.uint8) #127: glacier/ 0: shadow/ 64: rock/ 254: sea

    # save files
    fname = f"{base_name}.npy" 
    np.save(output_dir_images / fname, normalized_sar)
    np.save(output_dir_masks / fname, processed_mask)
    
    return 1 # each image is 1 file

def process_and_resize_files(file_list, raw_sar_dir, subset_name): # prevent leakage
    
    count = 0
    
    for zone_path in file_list:
        try:
            zone_filename = Path(zone_path).name
            
            # sar_image file name based on zones file
            sar_filename = zone_filename.replace('_zones', '')
            sar_path = raw_sar_dir / sar_filename

            zone_arr = cv2.imread(zone_path, cv2.IMREAD_GRAYSCALE)
            sar_arr = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            
            if zone_arr is None or sar_arr is None or zone_arr.shape != sar_arr.shape:
                continue

            base_name = Path(zone_path).stem 
            # resize + save
            count += resize_and_save_full_image(sar_arr, zone_arr, base_name, subset_name)
            
        except Exception as e:
            print(f"Action Failed: {zone_path} ({subset_name}): {e}")
    return count

all_zone_train_files = glob(os.path.join(RAW_ZONE_TRAIN_DIR, '*_zones.png'))
print(f"Initial Training Data Files: {len(all_zone_train_files)}")

train_files, val_files = train_test_split(
    all_zone_train_files, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED
)

train_images = process_and_resize_files(train_files, RAW_SAR_TRAIN_DIR, "train")
val_images = process_and_resize_files(val_files, RAW_SAR_TRAIN_DIR, "val")
all_zone_test_files = glob(os.path.join(RAW_ZONE_TEST_DIR, '*_zones.png'))
print(f"Initial Test Data Files: {len(all_zone_test_files)}")

test_images = process_and_resize_files(all_zone_test_files, RAW_SAR_TEST_DIR, "test")

print(f"training set: {train_images} images")
print(f"validation set: {val_images} images")
print(f"test set: {test_images} images")
