import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def noise_reduction(data:np.ndarray, threshold: float = 0.03) -> list:
    del_list = []
    for i in range(len(data)):
        if data[i, 1] >= 0:
            del_list.append(i)
            continue 

        if i > 0 and data[i, 0] - data[i-1, 0] > threshold: 
            del_list.append(i-1)
            continue 

        if i<len(data) - 3 and data[i+1, 0] - data[i, 0] < 0 and data[i+2, 0] - data[i+1, 0] > 0 and data[i+3, 0] - data[i+2, 0] <0:
            del_list.append(i+1)
            del_list.append(i+2)
            
    return del_list

def noise_reduction_2(data:np.ndarray, threshold: float = 0.03) -> list:
    del_list = []
    for i in range(len(data)- 4):
        if data[i+1, 0] - data[i, 0] < 0 and data[i+2, 0] - data[i+1, 0] > 0 and data[i+3, 0] - data[i+2, 0] <0:
            del_list.append(i+1)
            del_list.append(i+2)
            
    return del_list

def operation(data:pd.core.frame.DataFrame):
    data = data
    data_tmp = data.values 
    chopped_data_tmp = data_tmp[:, 16:18]

    del_list = noise_reduction(chopped_data_tmp)
    data = data.drop(del_list, axis=0)


class NoiseReducer():
    def __init__(self):
        self.data = None 
        self.data_values = None
        self.del_list = []
        self.del_list_2 = []
        self.result_data = None

    def read(self, PATH):
        self.data = pd.read_csv(PATH)
        self.data_values = self.data.values
        return 
    
    def save(self, PATH):
        self.result_data.to_csv(PATH, index=None)
        return 
    
    def operate(self, threshold: float=0.03):
        self.chopped_data_tmp = self.data_values[:, 16:18]
        self.del_list = noise_reduction(self.chopped_data_tmp)
        self.result_data = self.data.drop(self.del_list, axis=0)       
        return 
        


