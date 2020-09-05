import pandas as pd
import numpy as np
import random 

class Training_Algorithm:

    def __init__(self):
        self.attr = []


    #ASSUMING THE ID COLUMN IS YEETED 
    def ShuffleData(self, df: pd.DataFrame) ->pd.DataFrame: 
        TotalNumColumns = 0 
        InOrder = list() 
        for i in df: 
            TotalNumColumns += 1 
            InOrder.append(i)
        Num_Columns_To_Shuffle = TotalNumColumns * .1 
        temp = list() 
        for i in range(Num_Columns_To_Shuffle): 
            Col = Random.randint(0,len(InOrder))
            temp.append(InOrder[Col])
            InOrder.remove(InOrder[Col])
        for i in InOrder: 
            temp.append(i)
        string = ""
        count = 0 
        for i in temp:
            if count == len(temp): 
                string += i
                break
            string += i 
            string += ','
            count+=1 
            continue  
        return df1 = df[[string]
        

    def CrossValidation(self,df: pd.DataFrame) -> list():
        #Calculate the number of records to be sampled for testing 
        TestSize = len(df) * .1 
        #Count until we hit the number of records we want to sample 
        for i in range(TestSize): 
            #Set a value to be a random number from the dataset 
            TestValue = random.randint(0,len(df))
            #Append this row to a new dataframe
            df1.append(df.drop(df.Index[TestValue]))
        Temporary = list() 
        #Return the training and test set data 
        Temporary.append(df,df1)
        return temporary 

    def BinTestData(self, df: pd.DataFrame) -> list(): 
        #10 bins
        Binsize = 10          
        BinsInList = list()     
        #Calculate the number of records to be sampled for testing 
        TestSize = len(df) * .1 
        for i in range(Binsize):
            df1 = pd.DataFrame 
            #Count until we hit the number of records we want to sample 
            for i in range(TestSize):
                #Set a value to be a random number from the dataset 
                TestValue = random.randint(0,len(df))
                #Append this row to a new dataframe
                df1.append(df.drop(df.Index[TestValue]))
            BinsInList.append(df1)
        return BinsInList




    # take in dataset and calculate occurence of each class
    def calculateN(self, df: pd.DataFrame) -> dict:
        n = {}
        Class = len(df.columns)
        for i in range(len(df)):
            ClassValue = df.iloc[Class][i] 
            if ClassValue in n: 
               n[ClassValue] += 1 
               continue
            n[ClassValue] = 1        
        # init dict with keys = class names
        # iterate over all rows and increment the class associated with that row
        return n


    # take in n and create a new dict q that is each value / total rows
    def calculateQ(self, n: dict, TotalRows) -> dict:
        QValue = {} 
        for k in n.keys(): 
            QValue[k] = n[k] / TotalRows
        return QValue

    def calculateF(self, n: dict, df: pd.DataFrame) -> dict:
        f = {"class1": {"A1": 0, "A2": 0}, "class2": {"A1": 0, "A2": 0}}

        # init nested dict where first layer keys are classes and second layer keys are each possible attribute value
        # iterate over every column that is an attribute
            # iterate over every row
                # increment counter of the class x attribute value 
        # iterate over all values in nested dict
            # add 1 and divide by the count of examples in the class (n[class]) plus the total number of examples
            # i.e. (v + 1)/(n[class] + d)
        return f