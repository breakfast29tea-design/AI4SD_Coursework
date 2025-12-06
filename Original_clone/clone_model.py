import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation, Multiply, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Add, Multiply 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

# Load processed data
base = './amazon_processed/'
def load_npy_split(split):
    imgs = []
    masks = []

    img_dir = os.path.join(base, split, "images")
    mask_dir = os.path.join(base, split, "masks")

    fnames = sorted(os.listdir(img_dir))

    for fname in fnames:
        imgs.append(np.load(os.path.join(img_dir, fname)))
        masks.append(np.load(os.path.join(mask_dir, fname)))

    # Convert lists to arrays
    imgs = np.vstack(imgs)
    masks = np.vstack(masks)

    imgs = imgs.astype("float32")
    masks = masks.astype("float32")

    masks = (masks > 0).astype("float32")

    return imgs, masks

# Load
train_x, train_y = load_npy_split("training")
val_x, val_y = load_npy_split("validation")
test_x, test_y = load_npy_split("test")

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print(val_x.shape, val_y.shape)

def convBlock(input, filters, kernel, kernel_init='he_normal', act='relu',transpose=False):
    if transpose == False:
        #conv = ZeroPadding2D((1,1))(input)
        conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer =kernel_init)(input)
    else:
        #conv = ZeroPadding2D((1,1))(input)
        conv = Conv2DTranspose(filters, kernel, padding = 'same',kernel_initializer = kernel_init)(input)
        conv = Activation(act)(conv)
    return conv


def convBlock2(input, filters, kernel, kernel_init='he_normal', act='relu', transpose=False):
  if transpose == False:
    conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)
    conv = Activation(act)(conv)
    conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(conv)
    conv = Activation(act)(conv)
  else:
    conv = Conv2DTranspose(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)
    conv = Activation(act)(conv)
    conv = Conv2DTranspose(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(conv)
    conv = Activation(act)(conv)

  return conv

def attention_block(x, gating, inter_shape, drop_rate=0.25):
    
    # Find shape of inputs
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    # x vector input and processing
    theta_x = Conv2D(inter_shape, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', activation=None)(x)
    theta_x = MaxPooling2D((2,2))(theta_x)
    shape_theta_x = K.int_shape(theta_x)

    # gating signal ""
    phi_g = Conv2D(inter_shape, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', activation=None)(gating)
    shape_phi_g = K.int_shape(phi_g)

    # Add components
    concat_xg = Add()([phi_g, theta_x])
    act_xg = Activation('relu')(concat_xg)

    # Apply convolution
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', activation=None)(act_xg)

    # Apply sigmoid activation
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)

    # UpSample and resample to correct size
    upsample_psi = UpSampling2D(interpolation='bilinear', size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    
    # Correction for tf.broadcast_to error
    channels = shape_x[3]
    y_temp = tf.repeat(upsample_psi, repeats=channels, axis=-1)
    
    y = Multiply()([y_temp, x])

    return y

def UNetAM(trained_weights = None, input_size = (512,512,3), drop_rate = 0.25, lr=0.0001, filter_base=16):

    # Input layer
    inputs = Input(input_size) #batch_size=1

    ## Contraction phase
    conv = convBlock2(inputs, filter_base, 3)
    #conv0 = Dropout(drop_rate)(conv0)

    conv0 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv0 = convBlock2(conv0, 2 * filter_base, 3)

    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    conv1 = convBlock2(pool0, 4 * filter_base, 3)
    #conv1 = Dropout(drop_rate)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = convBlock2(pool1, 8 * filter_base, 3)
    #conv2 = Dropout(drop_rate)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = convBlock2(pool2, 16 * filter_base, 3)
    #conv3 = Dropout(drop_rate)(conv3)

    ## Expansion phase
    up4 = (Conv2DTranspose(8 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv3))
    merge4 = attention_block(conv2, conv3, 8 * filter_base, drop_rate) # Attention gate
    conv4 = Concatenate()([up4, merge4]) #concatenate([up4, merge4]) !!!
    conv4 = convBlock2(conv4, 8 * filter_base, 3)

    up5 = (Conv2DTranspose(4 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv4))
    merge5 = attention_block(conv1, conv4, 4 * filter_base, drop_rate) # Attention gate
    conv5 = Concatenate()([up5, merge5])
    conv5 = convBlock2(conv5, 4 * filter_base, 3)

    up6 = (Conv2DTranspose(2 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv5))
    merge6 = attention_block(conv0, conv5, 2 * filter_base, drop_rate) # Attention gate
    conv6 = Concatenate()([up6, merge6])
    conv6 = convBlock2(conv6, 2 * filter_base, 3)

    up7 = (Conv2DTranspose(1 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv6))
    merge7 = attention_block(conv, conv6, 1 * filter_base, drop_rate) # Attention gate
    conv7 = Concatenate()([up7, merge7])
    #conv7 = concatenate([up7, conv]) # will cover the last line!!!
    conv7 = convBlock2(conv7, 1 * filter_base, 3)

    ## Output layer
    out = convBlock(conv7, 1, 1, act='sigmoid')

    model = Model(inputs, out)
   
    if trained_weights != None:
    	model.load_weights(trained_weights)

    return model

UNetAM().summary()

model = UNetAM(input_size=(512,512,4), filter_base=16, lr=0.0005)


def iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def f1(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum(y_pred) - tp
    fn = tf.reduce_sum(y_true) - tp
    return (2*tp) / (2*tp + fp + fn + 1e-7)

model.compile(optimizer = Adam(learning_rate = 0.0005), 
              loss = BinaryCrossentropy(), 
              metrics = [iou, f1])

history = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=2,
    epochs=20 #60
)


loss, iou_score, f1_score = model.evaluate(test_x, test_y, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test IoU: {iou_score:.4f}")
print(f"Test F1 Score: {f1_score:.4f}")