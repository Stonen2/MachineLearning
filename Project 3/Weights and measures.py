# Demonstrate and explain how an element is propagated through 2 hidden layers
# Demonstrate weight updates occurring on a two-layer network
#Demonstrate the gradient calculations at the output for any network
#
import numpy as np
import pandas as pd
import math, random, copy
import TestData
import DataUtility
import NeuralNetwork
import matplotlib.pyplot as plt
import time 
import Performance

# this function batches data for training the NN, batch size is thie important input parmameter
def batch_input_data(X: np.ndarray, labels: np.ndarray, batch_size: int) -> list:
    batches = []
    # grabs indices of all data points to train on
    data_point_indices = list(range(X.shape[1]))
    # shuffles them
    random.shuffle(data_point_indices)
    # print(data_point_indices)
    # then batches them in a list of [batch, batch labels] pairs
    for i in range(math.ceil(X.shape[1]/batch_size)):
        if i == math.ceil(X.shape[1]/batch_size) - 1:
            batch_indices = data_point_indices
        else:
            batch_indices = data_point_indices[:batch_size]
            data_point_indices = data_point_indices[batch_size:]
        # print(batch_indices)
        # batch indices is an array, selecting all columns of indices in that array
        X_i = X[:, batch_indices]
        labels_i = labels[:, batch_indices]
        batches.append([X_i, labels_i])
    return batches







#######################################################################################################################################
#
#Show a sample model for the smallest of each Neural network 0,1,2

#######################################################################################################################################
def main(): 
    print("=================Video Part 3 STARTING=================")
    
    #Show the sample model for smalled of NN types
    #Soybean dataset on a single layered {Display the Weights, and the Ground truth and Estimate}
    data_sets = ["abalone","Cancer","glass","forestfires","soybean","machine"] 

    regression_data_set = {
        "soybean": False,
        "Cancer": False,
        "glass": False,
        "forestfires": True,
        "machine": True,
        "abalone": True
    }
    categorical_attribute_indices = {
        "soybean": [],
        "Cancer": [],
        "glass": [],
        "forestfires": [],
        "machine": [],
        "abalone": []
    }
    for bb in range(3): 
        for data_set in data_sets:
            if data_set != 'soybean':
                continue

            du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
            # ten fold data and labels is a list of [data, labels] pairs, where 
            # data and labels are numpy arrays:
            tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

            # execute driver for each of the ten folds
            for j in range(10):
                test_data, test_labels = copy.deepcopy(tenfold_data_and_labels[j])
                #Append all data folds to the training data set
                remaining_data = [x[0] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
                remaining_labels = [x[1] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
                #Store off a set of the remaining dataset 
                X = np.concatenate(remaining_data, axis=1) 
                #Store the remaining data set labels 
                labels = np.concatenate(remaining_labels, axis=1)
                #Test data print to the scren 
                regression = regression_data_set[data_set]
                #If the data set is a regression dataset
                if regression == True:
                    #The number of output nodes is 1 
                    output_size = 1
                #else it is a classification data set 
                else:
                    #Count the number of classes in the label data set 
                    output_size = du.CountClasses(labels)
                    #Get the test data labels in one hot encoding 
                    test_labels = du.ConvertLabels(test_labels, output_size)
                    #Get the Labels into a One hot encoding 
                    labels = du.ConvertLabels(labels, output_size)
                input_size = X.shape[0]

                ############# hyperparameters ################
                if bb == 0: 
                    hidden_layers = []
                elif bb == 1: 
                    hidden_layers = [input_size]
                else: 
                    hidden_layers = [input_size,input_size]
                
                # [] 0 Hidden Layers 
                # [input_size] 1 Layer 
                # [input_size, inpute_size ] 2 layers 
                learning_rate = .01
                momentum = 0
                batch_size = 20
                epochs = 500
                ##############################################
                NN = NeuralNetwork.NeuralNetwork(
                    input_size, hidden_layers, regression, output_size, learning_rate, momentum
                )
                print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ { data_set } $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                print("NUMBER OF HIDDEN LAYERS = " + str(bb))
                plt.ion()
                batches = batch_input_data(X, labels, batch_size)
                for i in range(epochs):
                    
                    for batch in batches:
                        X_i = batch[0]
                        labels_i = batch[1]
                        NN.set_input_data(X_i, labels_i)
                        NN.forward_pass()
                        NN.backpropagation_pass()
                    if i % 100 == 0:
                        plt.plot(NN.error_x, NN.error_y)
                        plt.draw()
                        plt.pause(0.00001)
                        plt.clf()
                
                Estimation_Values = NN.classify(test_data,test_labels)
                if regression == False: 
                    #Decode the One Hot encoding Value 
                    Estimation_Values = NN.PickLargest(Estimation_Values)
                    test_labels = NN.PickLargest(test_labels)
                else: 
                    Estimation_Values = Estimation_Values.tolist()
                    test_labels = test_labels.tolist() 
                    Estimation_Values = Estimation_Values[0]
                    test_labels =  test_labels[0]
                    #print(test_labels)
                    #time.sleep(10000)
                
                Per = Performance.Results()
                Estimat = Estimation_Values
                groun = test_labels
            
                Nice = Per.ConvertResultsDataStructure(groun, Estimat)
                print("THE GROUND VERSUS ESTIMATION:")
                print(Nice)
                print("\n")
                """
                hidden_layers = [input_size]
                learning_rate = .01
                momentum = 0
                batch_size = 20
                epochs = 500
                """
                Meta = list()
                #Meta Data order
                h1 = 0 
                h2 = 0 
                #The number of hidden layers is 0 
                if len(hidden_layers) == 0: 
                    #No hidden layers so 0 
                    h1 = 0 
                    h2 = 0 
                #THe number of hidden layers is 1 
                elif len(hidden_layers) == 1: 
                    #Set the number of nodes in the hidden layer 
                    h1 = hidden_layers[0]
                    #No layer so 0
                    h2 = 0 
                #The number of hidden layers is 2 
                else: 
                    #The number of nodes per hidden layer 
                    h1 = hidden_layers[0]
                    #The number of nodes per hidden layer 
                    h2 = hidden_layers[1]
                #The number of hidden layers 
                Meta.append(data_set)
                #The number of hidden layers
                Meta.append(len(hidden_layers))
                #Number of nodes in h1 
                Meta.append(h1)
                #Number of Nodes in h2 
                Meta.append(h2)
                #Learning Rate
                Meta.append(learning_rate) 
                #Momentum 
                Meta.append(momentum)
                #Batch Size 
                Meta.append(batch_size)
                #Epochs
                Meta.append(epochs)
                Per.StartLossFunction(regression,Nice,Meta)
            

    print("==================Video Part 3 ENDING==================")



main() 
