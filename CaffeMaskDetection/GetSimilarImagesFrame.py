
#################################################
# Imports and function definitions
#################################################

# Numpy for loading image feature vectors from file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path

# json for storing data in json file
import json

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial

# get path
from pathlib import Path

#################################################

#################################################
# This function reads from 'image_data.json' file
# Looks for a specific 'filename' value
# Returns the product id when product image names are matched
# So it is used to find product id based on the product image name
#################################################

#################################################
# This function;
# Reads all image feature vectors stored in /feature-vectors/*.npz
# Adds them all in Annoy Index
# Builds ANNOY index
# Calculates the nearest neighbors and image similarity metrics
# Stores image similarity scores with productID in a json file
#################################################
def cluster():
    # File path
    path = Path(__file__).parent.absolute().parent.absolute()

    # Defining data structures as empty dict
    file_index_to_file_name = {}
    file_index_to_file_vector = {}

    # Configuring annoy parameters
    dims = 1792
    n_nearest_neighbors = 5  # amount of neighbors
    trees = 15000 #10000

    # Reads all file names which stores feature vectors
    allfiles = os.listdir(os.path.sep.join([str(path), "FeatureVectors"]))

    # formerly, angular -> euclidean -> manhattan -> hamming -> dot
    t = AnnoyIndex(dims, metric='euclidean')

    for file_index, i in enumerate(allfiles):
        # Reads feature vectors and assigns them into the file_vector
        file_vector = np.loadtxt(os.path.sep.join([str(path), "FeatureVectors", i]))

        # Assigns file_name and feature_vectors
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector

        # Adds image feature vectors into annoy index
        t.add_item(file_index, file_vector)

    # Builds annoy index
    t.build(trees)

    # Array to hold nearest neighbor
    named_nearest_neighbors = []

    # Loops through all indexed items
    for i in file_index_to_file_name.keys():

        # only find comparison for the frame vector
        if (file_index_to_file_name[i] != "frame"):
            continue

        # Assigns master file_name, image feature vectors and product id values
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]

        # Calculates the nearest neighbors of the master item
        # Calculates for an additional neighbor, because first/closest neighbor is always original image
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors + 1)

        # Loops through the nearest neighbors of the master item
        for count, j in enumerate(nearest_neighbors):

            # because image will always be 100% identical with itself, skip the first entry
            if count == 0:
                continue

            # Assigns file_name, image feature vectors and product id values of the similar item
            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]

            # Calculates the similarity score of the similar item
            similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            # Appends master product id with the similarity score
            # and the product id of the similar items
            named_nearest_neighbors.append({
                'similarity': rounded_similarity,
                'master_name' : master_file_name,
                'similar_name' : neighbor_file_name})

        # if frame vector is found, end loop
        if (file_index_to_file_name[i] == "frame"):
            break


    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)

    #maybe just have it return the dictionary

    return named_nearest_neighbors

#cluster()

