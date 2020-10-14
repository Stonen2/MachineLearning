#Written by Nick Stone edited by Matteo Bjornsson 
#################################################################### MODULE COMMENTS ############################################################################
# This program will run the data processor which will create the data processor to change the missing value of the data sets and return a pandas data frame     #
# This program is meant to run every KNN algorithm on a single dataset and NOT every dataset that is given in the project.                                      #
# The data structure that will be passforward is a numpy array                                                                                                  #
##
#################################################################### MODULE COMMENTS ############################################################################

import pandas as pd
import numpy as np
import sys
import random 
import copy 
import math 
import DataProcessor 



class DataUtility: 
    def __init__(self, categorical_attribute_indices, regression_data_set):
        self.categorical_attribute_indices = categorical_attribute_indices
        self.regression_data_set = regression_data_set
        print("initializing the Data")     



    #Parameters: take in a data set and the name of a given data set 
    #Returns:  Return the new data set with all categorical values conveted 
    #Function: Convery all of the categorical features to a integrer or real value 
    def ConvertData(self,data_set_row, Name):
        #For each of the indexes in the data_set_row 
        for i in range(len(data_set_row)): 
            #if the value is a N or an n from the vote data cast to a 1 
            if data_set_row[i] == 'N' or data_set_row[i] == 'n': 
                #Conver the value to 1 
                data_set_row[i] = 1
            #If the value that we are taking in from the vote data is a y 
            if data_set_row[i] == 'Y' or data_set_row[i] == 'y': 
                #Set the value to be a 0 
                data_set_row[i] = 0 
            #If the data from the forest fire is jan 
            if data_set_row[i] == 'jan': 
                #Set the value to 0 
                data_set_row[i] = 0/11
            #If the data from the forest fire is feb
            if data_set_row[i] == 'feb' : 
                #Set the value to be the value of the month divided by the total number of months starting from 0 
                data_set_row[i] = 1/11
            #If the data from the forest fire is mar
            if data_set_row[i] == 'mar': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 2/11
            #If the data from the forest fire is apr
            if data_set_row[i] == 'apr': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 3/11
            #If the data from the forest fire is may
            if data_set_row[i] == 'may': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 4/11
            #If the data from the forest fire is jun
            if data_set_row[i] == 'jun': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 5/11
            #If the data from the forest fire is jul
            if data_set_row[i] == 'jul': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 6 /11
            #If the data from the forest fire is aug
            if data_set_row[i] == 'aug': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 7 /11
            #If the data from the forest fire is sep
            if data_set_row[i] == 'sep': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 8 /11
            #If the data from the forest fire is oct
            if data_set_row[i] == 'oct':
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 9 /11
            #If the data from the forest fire is nov
            if data_set_row[i] == 'nov': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 10/11
            #If the data from the forest fire is dec
            if data_set_row[i] == 'dec': 
                #Set the value to be the value of the month divided by the total number of months starting from 0
                data_set_row[i] = 11/11
            #If the day of the week is Monday
            if data_set_row[i] == 'mon' : 
                #Set the value to be 0  
                data_set_row[i] = 0/6
            #If the day of the week is Tuesday
            if data_set_row[i] == 'tue': 
                #Set the value to be the 1st day divide by 6 days 
                data_set_row[i] = 1/6
            #If the day of the week is Wednesdayu
            if data_set_row[i] == 'wed': 
                #Set the value to be the 2nd day divide by 6 days 
                data_set_row[i] = 2/6
            #If the day of the week is Thursday
            if data_set_row[i] == 'thu': 
                #Set the value to be the 3rd day divide by 6 days 
                data_set_row[i] = 3/6
            #If the day of the week is Friday
            if data_set_row[i] == 'fri': 
                #Set the value to be the 4th day divide by 6 days 
                data_set_row[i] = 4/6
            #If the day of the week is Saturday
            if data_set_row[i] == 'sat':
                #Set the value to be the 5th day divide by 6 days  
                data_set_row[i] = 5 /6
            #If the value is sunday 
            if data_set_row[i] == 'sun':
                #Set the value to 1  
                data_set_row[i] = 6 /6
            #If the value is male 
            if data_set_row[i] == 'M':
                #Set the value to be .5
                data_set_row[i] = 1 /2
            #If the value if female 
            if data_set_row[i] == 'F':
                #Set the value to be 1  
                data_set_row[i] = 2 /2
            #if the value is infant 
            if data_set_row[i] == 'I':
                #Set the value to 0  
                data_set_row[i] = 0  /2
        #Return the updated dataset 
        return data_set_row
    
    def StratifyTenFold(self, df: pd.DataFrame): 
        #Set the bin size to 10 
        Binsize = 10
        #Create a List of column names that are in the dataframe 
        columnHeaders = list(df.columns.values)
        Classes = list() 
        for row in range(len(df)): 
            if df.iloc[row][len(df.columns)-1] not in Classes: 
                Classes.append(df.iloc[row][len(df.columns)-1])
            continue 
        ClassOccurence = list() 
        for i in Classes: 
            occurence = 0 
            for j in range(len(df)): 
                if df.iloc[j][len(df.columns)-1] == i: 
                    occurence +=1 
            ClassOccurence.append(occurence)
        bins= [] 
        for i in range(Binsize):
            #Append the dataframe columns to the list created above 
            bins.append(pd.DataFrame(columns=columnHeaders))
        binnum = 0 
        for i in Classes: 
            binnum = random.randint(0,Binsize-1)
            for j in range(len(df)): 
                if df.iloc[j][len(df.columns)-1] == i: 
                    bins[binnum] = bins[binnum].append(df.iloc[j],ignore_index=True)
                    binnum += 1 
                    if binnum == 10: 
                        binnum = 0 
        #Return the list of Bins 
        for i in range(Binsize):
            bins[i] = bins[i].to_numpy()
        return bins 
            


        #Generate a list of every unique Class 
        #Generate the occurence of each Class 
        #Break down each class occurence into 10
        #Bin each value accordingly 



        

    def ReplaceMissing(self,df: pd.DataFrame):
        #length = 3
        #Create a dataprocessor object and convert the data in the csv and change all missing attribtues 
        Dp = DataProcessor.DataProcessor()
        #Start the process to change the integrity of the dataframe from within the data processor
        data = Dp.ReplaceMissingValue(df) 
        return data 

    def ConvertDatastructure(self,df: pd.DataFrame): 
        #Convert the given Dataframe to a numpy array 
        Numpy = df.to_numpy() 
        #Return the numpy array 
        return Numpy

    #Remove 10 % of the data to be used as tuning data and seperate them into a unique dataframe 
    def TuningData(self,df: pd.DataFrame):
        #Make a deep copy of the data frame that we are taking in 
        remaining_data = copy.deepcopy(df)
        #Set the number of records to be 10 % of the data set we are taking in 
        Records = int(len(df) * .1)
        #Make another copy of the data frame 
        tuning_data = copy.deepcopy(df)
        #Store a blank copy of the data frame 
        tuning_data = tuning_data[0:0]
        #Loop until we have extracted 10 % of the data set 
        for i in range(Records):
            #Randomly remove a random record from the data set 
            Random =  random.randint(0,len(remaining_data)-1)
            #Store the record at the given randomly assigned indexed
            rec = remaining_data.iloc[Random]
            #Add the record that we just generated to a dataframe 
            tuning_data = tuning_data.append(remaining_data.iloc[Random],ignore_index = True)
            #Drop the record from the overall total dataset
            remaining_data = remaining_data.drop(remaining_data.index[Random])
            #Reset the indexs 
            remaining_data.reset_index()
        #Return the tuning data set and the rest of the data set to the calling function 
        return tuning_data, remaining_data
        

    #Parameters: DataFrame
    #Returns: List of dataframes 
    #Function: Take in a dataframe and break dataframe into 10 similar sized sets and append each of these to a list to be returned 
    def BinTestData(self, df: pd.DataFrame) -> list(): 
        #Set the bin size to 10 
        Binsize = 10
        #Create a List of column names that are in the dataframe 
        columnHeaders = list(df.columns.values)
        #Create an empty list 
        bins = []
        #Loop through the size of the bins 
        for i in range(Binsize):
            #Append the dataframe columns to the list created above 
            bins.append(pd.DataFrame(columns=columnHeaders))
        #Set a list of all rows in the in the dataframe 
        dataIndices = list(range(len(df)))
        #Shuffle the data 
        random.shuffle(dataIndices)
        #Shuffle the count to 0 
        count = 0
        #For each of the indexs in the dataIndices 
        for index in dataIndices:
            #Set the bin number to count mod the bin size 
            binNumber = count % Binsize
            bins[binNumber] = bins[binNumber].append(df.iloc[index], ignore_index=True)
            #Increment count 
            count += 1
            #Go to the next 
            continue
        #Return the list of Bins 
        for i in range(Binsize):
            bins[i] = bins[i].to_numpy()
        return bins

    # this function takes in the name of a preprocessed data set and normalizes
    # all continuous attributes within that dataset to the range 0-1.
    def min_max_normalize_real_features(self, data_set: str) -> None:
        # read in processed dataset
        df = pd.read_csv(f"./ProcessedData/{data_set}.csv")
        # create new data frame to store normalized data
        normalized_df = pd.DataFrame()
        # keep track of which column index we are looking at
        index = -1
        headers = df.columns.values
        # iterate over all columns
        for col in headers:
            index += 1
            # check if the index is categorical or ground truth. in this case do not normalize
            if index in self.categorical_attribute_indices[data_set] or col == headers[index]:
                normalized_df[col] = df[col]
                continue
            # generate a normalized column and add it to the normalized dataframe
            min = df[col].min()
            max = df[col].max()
            if min == max:
                print(f"Column {col} deleted, all elements are the same.")
                continue
            normalized_df[col] = (df[col] - min)/(max - min)
        # save the new normalized dataset to file
        normalized_df.to_csv(f"./NormalizedData/{data_set}.csv", index=False)

    def get_tuning_data(self, data_set:str) -> np.ndarray:
        # read in data set
        df = pd.read_csv(f"./NormalizedData/{data_set}.csv")
        # extract data from dataset to tune parameters
        data_and_remainder = self.TuningData(df)
        # convert the tuning data set to numpy array
        tuning_data = data_and_remainder[0].to_numpy()
        return tuning_data

    # this function takes in experiment ready data and returns all forms of data required for the experiment 
    def generate_experiment_data(self, data_set: str)-> (list, np.ndarray, np.ndarray, list):
        # read in data set
        df = pd.read_csv(f"./NormalizedData/{data_set}.csv")
        # save the column labels
        headers = df.columns.values
        # extract data from dataset to tune parameters
        #tuning_data, remainder = self.TuningData(df)
        # convert the tuning data set to numpy array
        #tuning_data = tuning_data.to_numpy()
        # split the remaining data into 10 chunks for 10fold cros validation
        tenFolds = self.BinTestData(df)
        # save the full set as numpy array
        full_set = df.to_numpy()
        # return the headers, full set, tuning, and 10fold data
        return headers, full_set, tenFolds 

    # this function takes in experiment ready data and returns all forms of data required for the experiment 
    def generate_experiment_data_Categorical(self, data_set: str)-> (list, np.ndarray, np.ndarray, list):
        # read in data set
        df = pd.read_csv(f"./NormalizedData/{data_set}.csv")
        # save the column labels
        headers = df.columns.values
        # extract data from dataset to tune parameters
        #tuning_data, remainder = self.TuningData(df)
        # convert the tuning data set to numpy array
        #tuning_data = tuning_data.to_numpy()
        # split the remaining data into 10 chunks for 10fold cros validation
        tenFolds = self.StratifyTenFold(df)
        # save the full set as numpy array
        full_set = df.to_numpy()
        # return the headers, full set, tuning, and 10fold data
        return headers, full_set, tenFolds 




if __name__ == '__main__':

    categorical_attribute_indices = {
        "soybean": [],
        "Cancer": [],
        "glass": [],
        "forestfires": [],
        "machine": [],
        "abalone": []
    }
    Data_Sets = ["abalone","Cancer","glass","forestfires","soybean","machine"] 
    regression_data_set = {
        "soybean": True,
        "Cancer": True,
        "glass": True,
        "forestfires": True,
        "machine": True,
        "abalone": True
    }
    
    print("Testing the interface between pandas and numpy arrays")
    Vote_Data = "C:/Users/nston/Desktop/MachineLearning/Project 3/Cancer/Cancer.data"
    Glass_Data = ""
    Seg_Data = ""
    df = pd.read_csv(Vote_Data)
    print(df)
    Df1 = DataUtility(categorical_attribute_indices, regression_data_set)
    dfs = Df1.ReplaceMissing(df)
    for i in Data_Sets: 
        print(i)
        Df1.min_max_normalize_real_features(i)

    #print(dfs)
    # test = list() 
    #Tuning = Df1.StratifyTenFold(dfs)
    #for i in Tuning: 
  
  
