'''
   Copyright 2024 Karthik Srinivasan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from gradientDescentMeta import ztheta
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
from scipy.spatial.distance import braycurtis as bcd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm


class phenotype_constraint():
    def __init__(self, data, metadata, metadata_names, eta, num_steps, num_latents, alpha):
        self.data = data
        self.metadata = metadata
        self.metadata_names = metadata_names
        self.eta = eta
        self.num_steps = num_steps
        self.num_latents = num_latents
        self.alpha = alpha

        self.metadata_mean = np.mean(self.metadata, axis=0)
        self.metadata_std = np.std(self.metadata, axis=0)
        self.metadata_z_scored = (self.metadata - self.metadata_mean[np.newaxis, :])/self.metadata_std[np.newaxis, :]

        self.test_train_split()

        train_obj = ztheta(self.train_data, self.train_meta, self.eta, self.num_steps, self.num_latents, self.alpha, plot=True)

        if not train_obj.converged:
            raise RuntimeError('Gradient descent did not converge. Change learning parameters')        
        
        self.z_train = train_obj.z
        self.theta = train_obj.theta
        self.C = train_obj.C

        test_obj = ztheta(self.test_data, self.test_meta, self.eta, self.num_steps, self.num_latents, self.alpha, plot=True, theta_update=False, theta_initial=self.theta, C_initial = self.C)
        if not test_obj.converged:
            raise RuntimeError('Gradient descent did not converge. Change learning parameters')  
        self.z_test = test_obj.z
        self.fit_GMM()
        # self.create_in_silico_samples()

        

    def test_train_split(self):
        rng_idx = np.random.permutation(int(self.data.shape[0]))
        train_length = int(0.8*self.data.shape[0])
        train_data_idx = rng_idx[:train_length]
        test_data_idx = rng_idx[train_length:]

        self.test_data = self.data[test_data_idx]
        self.train_data = self.data[train_data_idx]

        self.test_meta = self.metadata_z_scored[test_data_idx]
        self.train_meta = self.metadata_z_scored[train_data_idx]


    def fit_GMM(self):
        '''
        Finds the number of clusters in the latent variables. 
        Fits the best Gaussian mixture model with the optimal number of clusters
        '''
        bic_list = []
        for i in range(1, 6):
            best_score = -np.inf
            best_model = None
            for _ in range(100):
                gm = GM(n_components=i).fit(self.z_train)
                current_score = gm.score(self.z_train)
                if current_score > best_score:
                    best_score = current_score
                    best_model = gm
            bic_list.append(best_model.bic(self.z_train))

        self.num_clusters = np.argsort(bic_list)[0] + 1

        print('Number of clusters in latents = {}'.format(self.num_clusters))
        plt.plot(np.arange(1,6), bic_list)
        plt.xticks(np.arange(1,6))
        plt.title('BIC vs no. of clusters')
        plt.xlabel('No. of clusters')
        plt.ylabel('BIC')
        plt.show()


        self.best_model = None
        best_score = -np.inf
        for _ in range(100):
            gmm = GM(n_components=self.num_clusters).fit(self.z_train)
            current_score = gmm.score(self.z_train)
            if current_score > best_score:
                best_score = current_score
                self.best_model = gmm    
    def create_in_silico_samples(self, num_samples=1e6):
        #Select the best model to sample
        gm = self.best_model
        gm_latents, _ = gm.sample(num_samples)
        
        #Create sampled data
        self.gm_data = np.exp(-np.matmul(gm_latents, self.theta))
        self.gm_data /= np.sum(self.gm_data, axis=1)[:, np.newaxis]
        self.gm_meta = np.matmul(gm_latents, self.C)
    
    def rel_BCD_vs_phenotype(self, phenotype, plot=True, return_constraint=False):
        '''
        Calculates the relative BCD when specified phenotype is constrained to be any particular value.
        '''
        def bin_array(array, num_bins):
            # Determine bin width
            array_min = np.min(array)
            array_max = np.max(array)
            bin_width = (array_max - array_min) / num_bins

            # Initialize bins
            bins = [[] for _ in range(num_bins)]

            # Bin the elements
            for i, value in enumerate(array):
                bin_index = min(int((value.item() - array_min) / bin_width), num_bins - 1)
                bins[bin_index].append(i)

            return bins


        self.create_in_silico_samples(5*1e5)
        #Get bins for constraining phenotype values
        bins = bin_array(self.gm_meta[:, self.metadata_names==phenotype], 50)

        #Average BCD in the data
        #Randomly sample pairs
        rng_idxs = np.random.default_rng().integers(len(self.gm_data), size=(100000, 2))
        rng_idxs = rng_idxs[np.where(rng_idxs[:, 0] != rng_idxs[:, 1])]

        rng_pairs = np.array([[self.gm_data[idxs[0]], self.gm_data[idxs[1]]] for idxs in rng_idxs])#list(itertools.combinations(self.gm_data, 2))[:100000]
        
        #Calculate bcd for the pairs
        mean_rng_bcd = np.mean([bcd(pair[0], pair[1]) for pair in rng_pairs])


        #Calculate 
        bcd_mean_mean_list = []
        bcd_std_list = []
        met_value_list = []
        for bin_idxs in bins:
            if len(bin_idxs) < 50:
                continue
            
            constrained_data = self.gm_data[bin_idxs, :]
            avg_met_value = self.gm_meta[bin_idxs, self.metadata_names==phenotype].mean()
            met_value_list.append(avg_met_value)

            bcd_mean_list = []
            for _ in range(25):
                rand_idxs = np.random.permutation(constrained_data.shape[0])[:min(len(bin_idxs), 100)]
                #Randomly sample 100 from the constrained data if there is less than 10 in a bin return Nan
                constrained_pairs = list(itertools.combinations(constrained_data[rand_idxs, :], 2))
                bcd_list = np.array([bcd(pair[0], pair[1]) for pair in constrained_pairs])/mean_rng_bcd
                bcd_mean = bcd_list.mean()
                bcd_mean_list.append(bcd_mean)
                # bcd_std = bcd_list.std()
            bcd_mean_mean_list.append(np.mean(bcd_mean_list))            
            bcd_std_list.append(np.std(bcd_mean_list))

        if plot:
            met_values_scaled = np.array(met_value_list)*self.metadata_std[self.metadata_names==phenotype] + self.metadata_mean[self.metadata_names==phenotype]
            x = np.linspace(min(met_values_scaled), max(met_values_scaled), 100)
            plt.plot(x, np.ones(100), ':', color='black')
            plt.errorbar(met_values_scaled, bcd_mean_mean_list, bcd_std_list, color='blue')

            plt.ylim([0,1.2])
            plt.yticks([0,0.5,1])
            plt.xlabel('{}'.format(phenotype))
            plt.ylabel('relative Bray-Curtis dissimilarity')
            # plt.show()

        if return_constraint:
            return 1-np.mean(bcd_mean_mean_list)

    def phenotype_constraint_list(self, num_cores=1):
        self.constraint_level_list = []
        for i in tqdm(range(10)):
            constraint_level = Parallel(n_jobs=num_cores)(
            delayed(self.rel_BCD_vs_phenotype)(phenotype, plot=False, return_constraint=True)
            for phenotype in self.metadata_names
            )
            self.constraint_level_list.append(constraint_level)
        
        self.contraint_level_list = np.array(self.constraint_level_list)
        self.constraint_level = np.mean(self.constraint_level_list, axis=0)
    
    def bcd_constrained_by_meta(self, phenotype_list, num_cores=1):
        '''
        This function predicts the composition constrained by the phenotypes in the list for each data point in the test data.
        For each test data point we can then compute the BCD between the predicted composition and the test data point.
        The list of BCDs for each test data point is returned.
        '''
        phen_test_data = self.test_meta[:, np.isin(self.metadata_names, phenotype_list)]
        
        def process_data_point(data_point, phen):
            counter = 0
            pred_comp_list = []  # List of predicted compositions
            while (len(pred_comp_list) < 2 and counter < 1000):  # Counter limit changed, it was missing.
                self.create_in_silico_samples()
                # print('in silico samples created')
                gm_meta_phen = self.gm_meta[:, np.isin(self.metadata_names, phenotype_list)]            
                within_range_mask = np.all(np.abs(gm_meta_phen - phen) <= 0.10 * np.abs(phen), axis=1)
                within_range_indices = np.where(within_range_mask)[0]
                if len(within_range_indices) != 0:
                    for pred in self.gm_data[within_range_indices]:
                        pred_comp_list.extend(pred)
                counter += 1

            if len(pred_comp_list) == 0:
                raise ValueError('No matching points found. Increase threshold or sample more points')

            pred_comp_list = np.array(pred_comp_list).reshape(-1, self.test_data.shape[1])
            pred_comp = np.mean(pred_comp_list, axis=0)

            # print('prediction computed')
            return bcd(pred_comp, data_point), len(pred_comp_list)
        
        # Parallelize the loop using joblib
        results = Parallel(n_jobs=num_cores)(
            delayed(process_data_point)(data_point, phen) 
            for data_point, phen in zip(self.test_data, phen_test_data)
        )
        
        # print('results computed')
        bcd_list = [result[0] for result in results]
        no_of_preds = [result[1] for result in results]
        
        return bcd_list, no_of_preds


    def rng_pairs_bcd(self):
        #Average BCD in the data
        #Randomly sample pairs
        rng_idxs = np.random.default_rng().integers(self.train_data.shape[0], size=(100000, 2))
        rng_idxs = rng_idxs[np.where(rng_idxs[:, 0] != rng_idxs[:, 1])]

        rng_pairs = np.array([[self.train_data[idxs[0]], self.train_data[idxs[1]]] for idxs in rng_idxs])#list(itertools.combinations(gm_data, 2))[:100000]
        
        #Calculate bcd for the pairs
        self.rng_bcd_list = np.array([bcd(pair[0], pair[1]) for pair in rng_pairs])


