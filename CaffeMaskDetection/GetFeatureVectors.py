#################################################
# Imports and function definitions
#################################################
# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For saving 'feature vectors' into a txt file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path

from pathlib import Path

#################################################

#################################################
# This function:
# Loads the JPEG image at the given path
# Decodes the JPEG image to a uint8 W X H X 3 tensor
# Resizes the image to 224 x 224 x 3 tensor
# Returns the pre processed image as 224 x 224 x 3 tensor
#################################################
def load_img(path):
    # Reads the image file and returns data type of string
    img = tf.io.read_file(path)

    # Decodes the image to W x H x 3 shape tensor with type of uint8
    img = tf.io.decode_jpeg(img, channels=3)

    # Resize the image to 224 x 244 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)

    # Converts the data type of uint8 to float32 by adding a new axis
    # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
    # This is required for the mobilenet model we are using
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    return img


#################################################
# This function:
# Loads the mobilenet model in TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_image_feature_vectors(module):
    path = Path(__file__).parent.absolute().parent.absolute()

    # Loops through all images in a local folder
    for filename in os.listdir(os.path.sep.join([str(path), "MaskedImages"])):  # assuming all are images
        npzFile = str(os.path.sep.join([str(path), "FeatureVectors", str(filename.split(".")[0] + ".npz")]))

        if os.path.exists(npzFile) and (str(filename) != "frame.jpg"):
            continue

        # Loads and pre-process the image
        img = load_img(os.path.sep.join([str(path), "MaskedImages", filename]))

        # Calculate the image feature vector of the img
        features = module(img)

        # Remove single-dimensional entries from the 'features' array
        feature_set = np.squeeze(features)

        # Saves the image feature vectors into a file for later use
        outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
        out_path = os.path.sep.join([str(path), "FeatureVectors", outfile_name])

        # Saves the 'feature_set' to a text file
        np.savetxt(out_path, feature_set, delimiter=',')


#get_image_feature_vectors()
