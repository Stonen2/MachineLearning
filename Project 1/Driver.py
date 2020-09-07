#Created by Nick Stone and Matteo Bjornsson 
#Created on 8/20/2020 
#################################################################### MODULE COMMENTS ####################################################################
##
##
#################################################################### MODULE COMMENTS ####################################################################
import pandas as pd 
from DataProcessor import DataProcessor
from TrainingAlgorithm import TrainingAlgorithm
from Results import Results 
from Classifier import Classifier

#Take in the Result data, The data frame of data, and the trial number and print to a file 
def WriteToAFile(Results,DataFrame,Trial):
    FileName = "Naive Bayes Results " + str(Trial) + ".csv"
    f = open(FileName, "w")
    for i in results: 
        f.write(i + "\n")
    df.to_csv(FileName, header=None, index=None, sep=' ', mode='a')
    f.close()




def main(): 
    #What trial number we are on 
    Trial = 0 
    #Which set of the data is being used to test 
    TestData = 0 
    print("Program Starting")
    VoteData = 'Vote_Data//Votes.data'
    IrisData = 'Iris_Data//iris.data'
    GlassData = 'Glass_Data//glass.data'
    CancerData = 'Breast_Cancer_Data//cancer.data'
    SoybeanData = 'Soybean_Data//soybean.data'
    
    ####################################################### MACHINE LEARNING PROCESS #####################################################
    dp = DataProcessor()
    df = pd.read_csv(SoybeanData) 
    #Return a clean dataframe with missing attributes taken care of 
    df = dp.StartProcess(df)
  
    ML = TrainingAlgorithm()
    #Dataframe without noise Its a list of 10 mostly equal dataframes
    NoNoiseDf = ML.BinTestData(df)
    #DataFrame with Noise 
    NoiseDf =  ML.ShuffleData(df)
    #Return a list of 10 mostly equal sized dataframes
    NoiseDf = ML.BinTestData(NoiseDf)
    #Make One dataframe to hold all of the other Training dataframes 
    TrainingDataFrame = pd.DataFrame()
    #Make One dataframe that is our test Dataframe 
    TestingDataFrame = NoNoiseDf[TestData] 
    for i in range(len(NoNoiseDf)):     
        if i == TestData: 
            continue 
        else: 
            #Append the training dataframe to one dataframe to send to the ML algorithm 
            TrainingDataFrame.append(NoNoiseDf[i])
    
    #Calculate the N value for the Training set
    TrainingN = ML.calculateN(TrainingDataFrame)
    #Calculate the Q value for the Training set
    print(TrainingN)
    TrainingQ = ML.calculateQ(TrainingN,len(TrainingDataFrame))
    #Calculate the F Matrix for the Training set
    print(TrainingQ)
    TrainingF = ML.calculateF(TrainingN,TrainingDataFrame)
    #Create a Classifier Object to classify our test set 
    print(TrainingF)
    ClassifierObj = Classifier(TrainingN,TrainingQ,TrainingF)
    #Reassign the testing dataframe to the dataframe that has our Machine learning classification guesses implemented 
    TestingDataFrame = ClassifierObj.classify(TestingDataFrame)
    
    #Get some statistics on the Machine learning 
    #Create a Results object
    Analysis = Results()
    #List to hold our stats
    Stats = list()  
    #Run the 0/1 Loss function on our results
    Stats = Analysis.ZeroOneLossFunctionStats(TestingDataFrame)
    #Run the F1 Loss function on our results 




    #Send the Data to a csv file for human checking and hyper parameter tuning 
    WRiteToAFile(Stats, TestingDataFrame,Trial)

    #Increment the Trial and Testdata Number and do it again 
    Trial+=1 
    TestData +=1








    print("Program Finish")



#Call the main function
main() 