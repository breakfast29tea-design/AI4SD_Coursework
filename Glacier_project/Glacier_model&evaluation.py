import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model 
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from typing import List, Tuple
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import *

BASE_PATH = Path("./AI4SD_Coursework")
PROCESSED_DATA_ROOT = BASE_PATH / "S2GLC_Processed_Full" 

IMAGE_SIZE = 512

def load_npy_split(subset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    imgs = []
    masks = []
    
    img_dir = PROCESSED_DATA_ROOT / subset_name / "images"
    mask_dir = PROCESSED_DATA_ROOT / subset_name / "masks"

    fnames = sorted(os.listdir(mask_dir)) 
    
    for fname in fnames:
        if not fname.endswith(".npy"):
            continue
            
        mask_path = mask_dir / fname
        img_path = img_dir / fname

        if not img_path.exists():
            continue
             
        imgs.append(np.load(img_path))
        masks.append(np.load(mask_path))
        
    imgs = np.array(imgs)
    imgs = np.expand_dims(imgs, axis=-1) 
    
    masks = np.array(masks)
    masks = np.expand_dims((masks > 0).astype("float32"), axis=-1) 

    return imgs, masks

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.12),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomContrast(0.2)
])

train_x, train_y = load_npy_split("train") 
val_x, val_y = load_npy_split("val") 
test_x, test_y = load_npy_split("test")

print(f"training set(SAR/mask): {train_x.shape} / {train_y.shape}")
print(f"validation set(SAR/mask): {val_x.shape} / {val_y.shape}")
print(f"test set(SAR/mask): {test_x.shape} / {test_y.shape}")

# Model Definition
def convBlock2(x, filters, kernel=3):
    x = Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    return x

# simplified attention gate
def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, kernel_size=1, strides=2, padding='same')(x)  
    phi_g   = Conv2D(inter_channel, kernel_size=1, padding='same')(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi = Conv2D(1, kernel_size=1, padding='same')(f)
    psi = Activation('sigmoid')(psi)
    psi_up = UpSampling2D(size=(2, 2))(psi)
    y = Multiply()([x, psi_up])
    return y

# simplified UNet
def UNetAM_3layer(input_size=(512, 512, 1), drop_rate = 0.3, lr=0.0005, filter_base=16):
    inputs = Input(input_size)
    c1 = convBlock2(inputs, filter_base)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = convBlock2(p1, filter_base * 2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = convBlock2(p2, filter_base * 4)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    bn = convBlock2(p3, filter_base * 8)
    
    u1 = UpSampling2D(size=(2, 2))(bn)
    u1 = Conv2D(filter_base * 4, 3, padding='same', kernel_initializer='he_normal')(u1)
    att1 = attention_block(c3, bn, filter_base * 4)
    m1 = Concatenate()([u1, att1])
    c4 = convBlock2(m1, filter_base * 4)
    u2 = UpSampling2D(size=(2, 2))(c4)
    u2 = Conv2D(filter_base * 2, 3, padding='same', kernel_initializer='he_normal')(u2)
    att2 = attention_block(c2, c4, filter_base * 2)
    m2 = Concatenate()([u2, att2])
    c5 = convBlock2(m2, filter_base * 2)
    u3 = UpSampling2D(size=(2, 2))(c5)
    u3 = Conv2D(filter_base * 1, 3, padding='same', kernel_initializer='he_normal')(u3)
    att3 = attention_block(c1, c5, filter_base * 1)
    m3 = Concatenate()([u3, att3])
    c6 = convBlock2(m3, filter_base * 1)
    outputs = Conv2D(1, 1, activation='sigmoid')(c6)

    model = Model(inputs, outputs)
    return model

model = UNetAM_3layer()
model.summary()

# Evaluation
def iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def f1(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))
    
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-7),
    loss=bce_dice_loss,
    metrics=[iou, f1]
)

early_stop = EarlyStopping(
    monitor='val_f1',
    patience=2,        
    verbose=1,
    mode='max',
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'glacier_SIM.h5',
    monitor='val_f1',
    verbose=1,
    save_best_only=True,
    mode='max'
)

history = model.fit(
    data_augmentation(train_x), #train_x, 
    train_y,
    validation_data=(val_x, val_y), 
    batch_size=8,
    epochs=30,
    callbacks=[early_stop, checkpoint]
    )

loss, iou_score, f1_score = model.evaluate(test_x, test_y, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"IoU: {iou_score:.4f}")
print(f"F1 Score: {f1_score:.4f}")