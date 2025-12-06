import os
import numpy as np
import rioxarray as rxr
from pathlib import Path
from typing import List

BASE = Path("./AMAZON")
OUTPUT_DIR_NAME = 'amazon_processed'
TRAIN_LIMIT = 250

def normalize_image(img_array: np.ndarray) -> np.ndarray:
    arr = img_array.astype(np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return arr - min_val
    return (arr - min_val) / (max_val - min_val)

def process_data_set(subset_name: str, limit: int = 0) -> List[np.ndarray]:

    full_path = BASE / subset_name
    file_paths = list(full_path.glob("*.tif")) 
    
    if limit > 0:
        file_paths = file_paths[:limit]

    processed_arrays = []
    
    for file_path in file_paths:
        arr = np.array(rxr.open_rasterio(file_path))
        
        if 'image' in subset_name.lower() or 'images' in subset_name.lower():
            arr = normalize_image(arr)
            arr = arr.astype(np.float32) 
            arr = arr.T #(C, H, W) -> (H, W, C)
            arr = arr.reshape(-1, 512, 512, 4)
        
        elif 'mask' in subset_name.lower() or 'masks' in subset_name.lower() or 'label' in subset_name.lower():
            arr = arr.T #(C, H, W) -> (H, W, C)
            arr = arr.reshape(-1, 512, 512, 1)

        processed_arrays.append(arr)
    
    return processed_arrays, [p.name for p in file_paths]

training_images, training_images_list = process_data_set('Training/image', limit=TRAIN_LIMIT)
training_masks, training_masks_list = process_data_set('Training/label', limit=TRAIN_LIMIT)

validation_images, validation_images_list = process_data_set('Validation/images')
validation_masks, validation_masks_list = process_data_set('Validation/masks')

test_images, test_images_list = process_data_set('Test/image')
test_masks, test_masks_list = process_data_set('Test/mask')

def save_data(data_list, name_list, root_dir, subset_name):

    target_path = Path(root_dir) / subset_name
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Saving {subset_name}: there are ({len(data_list)} files)')
    
    for data, name in zip(data_list, name_list):

        npy_filename = target_path / name.replace('.tif', '.npy')
        np.save(npy_filename, data)

output_root = Path(OUTPUT_DIR_NAME)
output_root.mkdir(exist_ok=True)

save_data(training_images, training_images_list, output_root / 'training', 'images')
save_data(training_masks, training_masks_list, output_root / 'training', 'masks')

save_data(validation_images, validation_images_list, output_root / 'validation', 'images')
save_data(validation_masks, validation_masks_list, output_root / 'validation', 'masks')

save_data(test_images, test_images_list, output_root / 'test', 'images')
save_data(test_masks, test_masks_list, output_root / 'test', 'masks')

print(f"\n All files saved in '{OUTPUT_DIR_NAME}' ")