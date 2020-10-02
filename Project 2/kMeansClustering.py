#Written By 
#################################################################### MODULE COMMENTS ############################################################################
#################################################################### MODULE COMMENTS ############################################################################
import numpy as np
import DataUtility, kNN, Results
import copy, random


class kMeansClustering:
    
    def __init__(self,
        # number of clusters
        kValue: int,
        # data to cluster
        dataSet: np.ndarray,
        # 'mixed', 'categorical', or 'real' data set
        data_type: str,
        # list of integers representing categorical feature column indices
        categorical_features: list,
        # True if the data set is a regression data set
        regression_data_set: bool,
        # weight for real value in distance metric
        alpha: int,
        # weight for categorical value in distance metric
        beta: int,
        # bin width for gaussian kernel smoother
        h: float,
        # dimensionality of data set (# features)
        d: int):

        # create a Nearest Neighbor object to single nearest neighbor to input data point
        self.nn = kNN.kNN(1, data_type, categorical_features, regression_data_set, alpha, beta, h, d)
        self.categorical_features = categorical_features
        # save which features are real as well by deleting categorical indices from a new list
        real_features = list(range(d))
        for i in categorical_features:
            real_features.remove(i)
        self.real_features = real_features
        self.kValue = kValue
        self.dataSet = dataSet
        # dimensionality of data set
        self.d = d

    # randomly generate kvalue centroids by randomly generating an appropriate value per feature
    def create_random_centroids(self) -> np.ndarray:
        # first save all the unique values in the categorical features
        feature_values = [None] * self.d
        for i in self.categorical_features:
            # print(self.dataSet[:,i])
            feature_values[i] = np.unique(self.dataSet[:,i]).tolist()
            # print("distinct features:", feature_values[i])
        # create a blank list to append new points to
        points = []
        for k in range(self.kValue):
            # create a blank new point with the same dimensionality as data set
            new_point = [None] * self.d
            # assign categorical features first as a random selection of observed features per attribute
            for l in self.categorical_features:
                new_point[l] = random.choice(feature_values[l])
            # assign real value between 0 and 1 (as all real values are 0-1 normalized)
            for m in self.real_features:
                new_point[m] = random.uniform(0,1)
            # append the new point to the points list
            points.append(new_point)
        # return all points as a numpy array    
        return np.array(points)

    # find the nearest centroid to given sample, return the centroid index
    def closest_centroid_to_point(self, point: list, centroids: np.ndarray) -> list:
        # use the knn get_neighor class method to find the closest centroid 
        centroid = self.nn.get_k_neighbors(centroids, point, k=1)
        # return the centroid index, element 1 of [distance, index, response var]
        return centroid[1]
    
    # assign each data point in data set to the nearest centroid. This is stored in an array
    # as an integer representing the centroid index at the index of the point belonging to it. 
    def assign_all_points_to_closest_centroid(self, centriods: np.ndarray, data: np.ndarray) -> list:
        centroid_assignments = []
        # for each data point
        for i in range(len(data)):
            x = data[i].tolist()[:-1]
            # store the index of the centroid at the index corresponding to the sample position
            centroid_assignments[i] = self.closest_centroid_to_point(x, centriods)
        # return the list of indices
        return centroid_assignments

    def update_centroid_positions(self, centroids: np.ndarray, centroid_assignments: list, data: np.ndarray) -> np.ndarray:
        #TODO: write centroid update method (drop categorical values?)
        return centroids

    def generate_cluster_centroids(self):
        centroids = self.create_random_centroids()
        first_assignment = self.assign_all_points_to_closest_centroid(centroids, self.dataSet)
        updated_centroids = self.update_centroid_positions(centroids, first_assignment, self.dataSet)
        count = 0
        while True:
            second_assignment = self.assign_all_points_to_closest_centroid(updated_centroids, self.dataSet)
            updated_centroids = self.update_centroid_positions(updated_centroids, second_assignment, self.dataSet)
            count += 1
            if first_assignment == second_assignment or count > self.itermax:
                break
            first_assignment = second_assignment
        return updated_centroids

if __name__ == '__main__':
    categorical_attribute_indices = {
        "segmentation": [],
        "vote": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "glass": [],
        "fire": [0,1,2,3],
        "machine": [0,1],
        "abalone": [0]
    }

    regression_data_set = {
        "segmentation": False,
        "vote": False,
        "glass": False,
        "fire": True,
        "machine": True,
        "abalone": True
    }

    feature_data_types = {
        "segmentation": 'real',
        "vote": 'categorical',
        "glass": 'real',
        "fire": 'mixed',
        "machine": 'mixed',
        "abalone": 'mixed'
    }

    data_sets = ["segmentation", "vote", "glass", "fire", "machine", "abalone"]

    regression = [x for x in data_sets if regression_data_set[x]]

    for i in range(1):
        data_set = "vote"

        print("Data set: ", data_set)
        du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
        headers, full_set, tuning_data, tenFolds = du.generate_experiment_data(data_set)
        # print("headers: ", headers, "\n", "tuning data: \n",tuning_data)
        test = copy.deepcopy(tenFolds[0])
        training = np.concatenate(tenFolds[1:])

        d = len(headers)-1
        kMC = kMeansClustering(kValue=d, dataSet=training, categorical_features=categorical_attribute_indices[data_set], d=d)
        print(kMC.generateClusterPoints())