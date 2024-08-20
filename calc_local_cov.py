import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gradientDescentMeta import ztheta
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
from scipy.spatial.distance import braycurtis as bcd
import itertools
from phenotype_constraint import phenotype_constraint
import pickle
from joblib import Parallel, delayed
import json
import time
import glob
from tqdm import tqdm

def first_non_zero(arr):
    for i, elem in enumerate(arr):
        if elem != 0:
            return elem
    return None



def bacteria_phenotype_cov(x, phen, min_num_samples=50, timer_limit=None):
    start_time = time.time()
    phenotype_idx = [np.where(obj.metadata_names == phenotype)[0][0] for phenotype in phen]

    list_data_distances = np.sort([bcd(x, y) for y in obj.data])
    smallest_data_distance = first_non_zero(list_data_distances)

    list_gm_distances = np.array([bcd(x, y) for y in obj.gm_data])
    closest_gm_data_idx = np.argsort(list_gm_distances)

    sorted_gm_distances = list_gm_distances[closest_gm_data_idx]
    sorted_gm_data = np.copy(obj.gm_data)[closest_gm_data_idx]
    sorted_gm_meta = np.copy(obj.gm_meta)[closest_gm_data_idx]

    closest_gm_data = np.copy(sorted_gm_data)[sorted_gm_distances < smallest_data_distance]
    closest_gm_meta = np.copy(sorted_gm_meta)[sorted_gm_distances < smallest_data_distance]

    while closest_gm_data.shape[0] < min_num_samples:
        obj.create_in_silico_samples(1e5)
        list_gm_distances = np.array([bcd(x, y) for y in obj.gm_data])
        closest_gm_data_idx = np.argsort(list_gm_distances)

        sorted_gm_distances = list_gm_distances[closest_gm_data_idx]
        sorted_gm_data = np.copy(obj.gm_data)[closest_gm_data_idx]
        sorted_gm_meta = np.copy(obj.gm_meta)[closest_gm_data_idx]

        closest_gm_data_new = np.copy(sorted_gm_data)[sorted_gm_distances < smallest_data_distance]
        closest_gm_meta_new = np.copy(sorted_gm_meta)[sorted_gm_distances < smallest_data_distance]

        closest_gm_data = np.vstack((closest_gm_data, closest_gm_data_new))
        closest_gm_meta = np.vstack((closest_gm_meta, closest_gm_meta_new))

        current_time = time.time()  # Get the current time
        elapsed_time = current_time - start_time  # Calculate elapsed time
        
        # Check if the elapsed time exceeds the maximum duration
        if timer_limit is not None:
            if elapsed_time > 60*timer_limit:
                return {phen[i]: np.nan for i in range(len(phen))}
    
    closest_gm_data = closest_gm_data[:min_num_samples]
    closest_gm_meta = closest_gm_meta[:min_num_samples]

    # print(closest_gm_data.shape[0])

    cov = np.corrcoef(closest_gm_data, closest_gm_meta[:, phenotype_idx], rowvar=False)[:-len(phenotype_idx), -len(phenotype_idx):]

    assert cov.shape[0] == obj.gm_data.shape[1]
    assert cov.shape[1] == len(phen)
    assert closest_gm_data.shape[0] == min_num_samples

    output = {phen[i]: cov[:, i].tolist() for i in range(len(phen))}

    return output


# Train phenotype_constraint object or load pretrained object
with open("/home/ks2823/palmer_scratch/my_object.pkl", "rb") as f:
    obj = pickle.load(f)

print("Loaded object")

covariances_list = []
phenotype_of_interest = np.copy(obj.metadata_names)
data = obj.data  

def process_data_point(x, phenotype_of_interest):
    return bacteria_phenotype_cov(x, phenotype_of_interest, timer_limit=10)

print("Starting computation")
# Parallel computation
covariances_list = Parallel(n_jobs=-1)(
    delayed(process_data_point)(x, phenotype_of_interest) for x in data
)

print("Writing to file")
# Save results to file
file_name = 'data_files/all_phens_covs.json'
with open(file_name, 'w') as json_file:
    json.dump(covariances_list, json_file)

print(f"Processing completed at {time.time()}.")
