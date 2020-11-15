import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import matplotlib.pyplot as plt

class Particle:
    '''
    This class represents a particle in the swarm. Each particle maintains a 
    current position and velocity as well as a running record of its personal best
    position and fitness. 
    '''

    def __init__(self, position_range: float, velocity_range: float, layers: list):
        '''
        :param position_range: float. indicates the bounds for the random uniform
        distribution used to generate initial positions.
        :param velocity_range: float. same as above but for velocity.
        :param layers: list. Expected format: [input nodes, h1_nodes, h2_nodes, output_nodes]
        h1 and h2 can be missing but there must be an input and output: [input, output]
        '''
        # the total weights are represented by the number of weights needed to be 
        # estimated for the NN, which is the sum of each layer multiplied by the next layer
        total_weights = 0
        for i in range(len(layers)-1):
            total_weights += layers[i] * layers[i+1]
        # initialize each weight from a random uniform distribution
        self.position = np.random.uniform(-position_range, position_range, total_weights)
        # same for velocity
        self.velocity = np.random.uniform(-velocity_range, velocity_range, total_weights)
        # we are minimizing our fitness, so it is initialized as pos infinity
        self.fitness = float('inf')
        # current best is initial best
        self.pbest_fitness = self.fitness
        self.pbest_position = None


class PSO:
    '''
    this is the driver class for the PSO. For the given number of iterations or stopping condition
    it will update particle positions and velocities and track the global best position
    which is the position with the best fitness so far. 
    '''

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list, pop_size: int, NN:NeuralNetwork):
        #init general population 
            # random weight values, weight matrix is numpy array, matches network architecture
            # use similar weight init function as from NN

        # INIT BOTH *POSITION* (WEIGHTS) AND *VELOCITY*

        self.NN = NN
        # dictates the range of possible initial positions and velocities
        self.position_range = 10
        self.velocity_range = 1
        # store the list of nodes per layer
        self.layers = layers
        self.pop_size = pop_size

        # generate the swarm population
        self.population = [] 
        for i in range(pop_size):
            self.population.append(Particle(self.position_range, self.velocity_range, layers))
        # track the global best fitness. Initialized as inf as this is a min probelem
        self.gbest_fitness = float('inf')
        self.gbest_position = None
        # number of iterations 
        self.max_t = 1000

        #HYPERPARAMETERS:
        self.omega = .9
        self.c1 = .3
        self.c2 = 1.2
        self.vmax = 10
        # fitness plotting:
        self.fitness_plot = []

    def swarm_diversity(self) -> float:
        pass

    def swarm_fitness(self) -> float:
        # plot this over time to determine convergence?
        pass

    def update_position_and_velocity(self):
        # iterate over each particle
            # update v and x using equations from class
            # x_(t+1) = x_t + v_(t+1)
            # v_(t+1) = w*v_t + c1*r1*(pb_t - x_t) + c2*r2*(gb_t - x_t)
        for p in self.population:
            # assign variables to improve readability
            v = p.velocity
            w = self.omega
            c1 = self.c1
            r1 = random.uniform(0,1)
            c2 = self.c2
            r2 = random.uniform(0,1)
            pb = p.pbest_position
            gb = self.gbest_position
            x = p.position

            # calculate the new velocity
            new_v = w*v + c1*r1*(pb - x) + c2*r2*(gb - x)
            # update the new velocity and position
            p.velocity = new_v
            p.position += new_v

    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def update_fitness(self) -> None:
    # for all particles, this method applies the individual's weights to the NN, 
    # feeds data set through and sets the fitness to the error of forward pass

        layers = self.layers
        for p in self.population:
            weights = [None] * len(layers)
            weights[0] = []
            position = copy.copy(p.position)
            # transform the flat position vector into a list of weight matrices
            # for the neural network
            for i in range(len(layers)-1):
                l = layers[i] * layers[i+1]
                w = position[:l]
                position = position[l:]
                weights[i+1] = w.reshape(layers[i+1], layers[i])
            # run the dataset through the NN with the particle's weights to get fitness
            fitness = self.NN.fitness(weights)
            # update personal best
            if p.pbest_fitness > fitness:
                p.pbest_fitness = fitness
                p.pbest_position = p.position
            # update global best
            if self.gbest_fitness > fitness:
                self.gbest_fitness = fitness
                self.gbest_position = p.position
            # update particle fitness
            p.fitness = fitness

        # track global best over time each iteration
        self.fitness_plot.append(self.gbest_fitness)


    ####################################
    # driver method
    ####################################
    # initialize
    # until convergence of global best:
        # update V and X for each swarm memeber and eval fitness

if __name__ == '__main__':

    headers = ["Data set", "Hidden Layers", "h1 nodes", "h2 nodes", "learning rate", "momentum", "batch size", "batches", "epochs", "loss1", "loss2"]
    filename = 'PSO_test.csv'

    Per = Performance.Results()
    Per.PipeToFile([], headers, filename)

    data_sets = ["soybean", "glass", "abalone","Cancer","forestfires", "machine"] 

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

    tuned_0_hl = {
        "soybean": {
            "learning_rate": .001,
            "batch_count": 5,
            "epoch": 5000,
            "hidden_layer": []
        },
        "Cancer": {
            "learning_rate": .00001,
            "batch_count": 20,
            "epoch": 10000,
            "hidden_layer": []
        },
        "glass": {
            "learning_rate": .1,
            "batch_count": 5,
            "epoch": 10000,
            "hidden_layer": []
        },
        "forestfires": {
            "learning_rate": .00001,
            "batch_count": 10,
            "epoch": 10000,
            "hidden_layer": []
        },
        "machine": {
            "learning_rate": .1,
            "batch_count": 5,
            "epoch": 10000,
            "hidden_layer": []
        },
        "abalone": {
            "learning_rate": .01,
            "batch_count": 10,
            "epoch": 10000,
            "hidden_layer": []
        }
    }

    tuned_1_hl = {
        "soybean": {
            "learning_rate": .001,
            "batch_count": 10,
            "epoch": 10000,
            "hidden_layer": [7]
        },
        "Cancer": {
            "learning_rate": .000001,
            "batch_count": 5,
            "epoch": 500000,
            "hidden_layer": [4]
        },
        "glass": {
            "learning_rate": .001,
            "batch_count": 10,
            "epoch": 10000,
            "hidden_layer": [8]
        },
        "forestfires": {
            "learning_rate": .00001,
            "batch_count": 5,
            "epoch": 50000,
            "hidden_layer": [8]
        },
        "machine": {
            "learning_rate": .001,
            "batch_count": 5,
            "epoch": 10000,
            "hidden_layer": [4]
        },
        "abalone": {
            "learning_rate": .01,
            "batch_count": 5,
            "epoch": 10000,
            "hidden_layer": [8]
        }
    }

    tuned_2_hl = {
        "soybean": {
            "learning_rate": .001,
            "batch_count": 5,
            "epoch": 50000,
            "hidden_layer": [7,12]
        },
        "Cancer": {
            "learning_rate": .00000001,
            "batch_count": 5,
            "epoch": 500000,
            "hidden_layer": [4,4]
        },
        "glass": {
            "learning_rate": .001,
            "batch_count": 5,
            "epoch": 10000,
            "hidden_layer": [8,6]
        },
        "forestfires": {
            "learning_rate": .0001,
            "batch_count": 10,
            "epoch": 50000,
            "hidden_layer": [8,8]
        },
        "machine": {
            "learning_rate": .001,
            "batch_count": 5,
            "epoch": 10000,
            "hidden_layer": [7,2]
        },
        "abalone": {
            "learning_rate": .001,
            "batch_count": 10,
            "epoch": 5000,
            "hidden_layer": [6,8]
        }
    }
    for data_set in data_sets:
        if data_set != "machine": continue
        for j in range(1):
            
            du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
            # ten fold data and labels is a list of [data, labels] pairs, where 
            # data and labels are numpy arrays:
            tenfold_data_and_labels = du.Dataset_and_Labels(data_set)
            test_data, test_labels = copy.deepcopy(tenfold_data_and_labels[j])
            #Append all data folds to the training data set
            remaining_data = [x[0] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
            remaining_labels = [y[1] for i, y in enumerate(tenfold_data_and_labels) if i!=j]
            #Store off a set of the remaining dataset 
            X = np.concatenate(remaining_data, axis=1) 
            #Store the remaining data set labels 
            labels = np.concatenate(remaining_labels, axis=1)
            print(data_set, "training data prepared")
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

            data_set_size = X.shape[1] + test_data.shape[1]

            tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]
            for i in range(1):
                i=2
                hidden_layers = tuned_parameters[i]["hidden_layer"]
                ############################## new code for PSO start ##################################
                layers = [input_size] + hidden_layers + [output_size]

                nn = NeuralNetwork(input_size, hidden_layers, regression, output_size)
                nn.set_input_data(X,labels)

                pso = PSO(layers, 100, nn)

                plt.ion
                for j in range(40):
                    pso.update_fitness()
                    pso.update_position_and_velocity()
                    print("particle 1 position and velocity: \nposition:\n", pso.population[1].position, '\nvelocity:\n', pso.population[1].velocity)
                    print("particle 1 fitness: ", pso.population[1].fitness)
                    print("global best fitness:", pso.gbest_fitness)
                    plt.plot(list(range(len(pso.fitness_plot))), pso.fitness_plot)
                    plt.draw()
                    plt.pause(0.00001)
                    plt.clf()
                ################################# new code for PSO end ###################################
                
                Estimation_Values = pso.NN.classify(test_data,test_labels)
                if regression == False: 
                    #Decode the One Hot encoding Value 
                    Estimation_Values = pso.NN.PickLargest(Estimation_Values)
                    test_labels_list = pso.NN.PickLargest(test_labels)
                    # print("ESTiMATION VALUES BY GIVEN INDEX (CLASS GUESS) ")
                    # print(Estimation_Values)
                else: 
                    Estimation_Values = Estimation_Values.tolist()
                    test_labels_list = test_labels.tolist()[0]
                    Estimation_Values = Estimation_Values[0]
                
                Estimat = Estimation_Values
                groun = test_labels_list
                

                Nice = Per.ConvertResultsDataStructure(groun, Estimat)
                # print("THE GROUND VERSUS ESTIMATION:")
                # print(Nice)
                """
                hidden_layers = [input_size]
                learning_rate = .01
                momentum = 0
                batch_size = 20
                epochs = 500
                """
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

                #[data set, number of h layers, node per h 1, nodes per h2, learning rate, momentum, batch size, number batches, number epochs]
                Meta = [data_set, len(hidden_layers), h1, h2]
                Per.StartLossFunction(regression, Nice, Meta, filename)
