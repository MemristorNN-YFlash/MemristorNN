import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import datasets
import pandas as pd


class sets:
    def __init__(self,x,y, v1, v2, test_size=0.25, rnd_state=42):
        self.xtrain_set = 0
        self.ytrain_set = 0
        self.xtest_set = 0
        self.ytest_set = 0
        self.v1 = v1
        self.v2 = v2
        self.x = x
        self.y = y
        self.rnd_state = rnd_state
        self.test_size =test_size
        self.create_sets()
    
    def create_sets(self):
        self.x = self.norma_xy(self.x)
        self.xtrain_set,self.xtest_set, self.ytrain_set, self.ytest_set = train_test_split(self.x, self.y.transpose(), 
                                                                                           test_size= self.test_size)
                                                                                
    def norma_xy(self,df):
        ## Input dataframe - Output normalized between V_1 V_2 df
         return (self.v2 - self.v1)*normalize(df)
        
    def return_classes(self):
        return Set(self.xtrain_set, 0), Set(self.xtest_set, 0), Set(self.ytrain_set, output = 3),Set(self.ytest_set, output= 3)

class Set:
    def __init__(self,vec ,output = 0):
        self.num_of_inputs = vec.shape[0]
        self.current_entry = 0
        self.vec = vec
        self.yvec = np.zeros([output, self.num_of_inputs])
        if output != 0:
            self.make_output_vec()

        
    
    def make_output_vec(self):
        i = 0
        for i in range(self.num_of_inputs):
            self.yvec[self.vec[i]-1,i] =1
        self.vec = self.yvec.transpose()
    
    def next_entry(self):
        if self.current_entry + 1 <self.num_of_inputs:
            self.current_entry = self.current_entry + 1
        else:
            self.reset_entry()
            return True
        return self.vec[self.current_entry, :]
    
    def current(self):
        m = self.vec[self.current_entry,:]
        return m

    def reset_entry(self):
        self.current_entry = 0