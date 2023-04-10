import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import interpolate

def noise_reduction(data:np.ndarray, threshold: float = 0.03, chop: float = -1) -> list:
    del_list = []
    for i in range(len(data)):
        if data[i, 1] >= chop:
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
        self.current_data = None
        self.current_data_values = None
        self.data_values = None
        self.indexed_data_values = None 
        self.del_list = []
        self.del_list_stack = []
        self.stack = []
        self.stack_value = []
        self.stack_index = -1

    def read(self, PATH):
        self.data = pd.read_csv(PATH)
        self.current_data = self.data 
        self.data_values = self.data.values[:, 16:18]
        index_array = np.arange(0, self.data_values.shape[0] , 1, dtype=int).T 
        index_array = index_array.reshape((len(index_array), 1))
        self.current_data_values = np.concatenate((self.data_values, index_array), axis=1)
        self.indexed_data_values = self.current_data_values
        self.stack.append(self.current_data)
        self.stack_value.append(self.current_data_values)
        self.stack_index += 1 
        print("read finished")
        return 
    
    def save(self, PATH):
        self.current_data.to_csv(PATH, index=None)
        return 
    
    def operate(self, threshold: float=0.03, chop: float=-0.1):
        self.chopped_data_tmp = self.data_values[:, 16:18]
        self.del_list = noise_reduction(self.chopped_data_tmp)
        self.result_data = self.data.drop(self.del_list, axis=0)       
        return 
        
    def show(self):
        x_val = self.current_data_values[:, 0].squeeze()
        y_val = self.current_data_values[:, 1].squeeze()

        function = interpolate.interp1d(x_val, y_val, kind='linear')
        x_new = np.linspace(np.min(x_val), np.max(x_val), num=100, endpoint=True, dtype=float)
        try:
            y_new = function(x_new)
            plt.plot(x_val, y_val, 'o', x_new, y_new, '-')
            plt.plot(x_val, y_val, 'o')
            plt.legend(['Point', 'linear'])
            plt.xlabel('distance')
            plt.ylabel('depth')
            plt.show()

        except:
            plt.plot(x_val, y_val, 'o')
            plt.legend(['Point'])
            plt.xlabel('distance')
            plt.ylabel('depth')
            plt.show()

        return

    def chop(self, x_min=-100, x_max=100, y_min=-100, y_max=100):
        for i in range(self.current_data_values.shape[0]):
            if self.current_data_values[i, 0] < x_min or self.current_data_values[i, 0] > x_max:
                self.del_list.append(self.current_data_values[i,2])
                continue 
            
            if self.current_data_values[i, 1] < y_min or self.current_data_values[i, 1] > y_max:
                self.del_list.append(self.current_data_values[i,2])
        
        self.del_list = list(set(self.del_list))
        self.current_data = self.data.drop(self.del_list, axis=0)
        self.current_data_values = np.delete(self.indexed_data_values, self.del_list, axis=0)
        self.stack.append(self.current_data)
        self.stack_value.append(self.current_data_values)
        self.stack_index += 1
        return 
 
    def noise_reduction(self, threshold:float = 0.03):
        for i in range(self.current_data_values.shape[0]):
            if i > 0 and self.current_data_values[i, 0] - self.current_data_values[i-1, 0] > threshold: 
                self.del_list.append(self.current_data_values[i, 2])
                continue 

            if i<len(self.current_data_values) - 3 and self.current_data_values[i+1, 0] - self.current_data_values[i, 0] < 0 and self.current_data_values[i+2, 0] - self.current_data_values[i+1, 0] > 0 and self.current_data_values[i+3, 0] - self.current_data_values[i+2, 0] <0:
                self.del_list.append(self.current_data_values[i+1, 2])
                self.del_list.append(self.current_data_values[i+2, 2])

        self.del_list = list(set(self.del_list))
        self.current_data = self.data.drop(self.del_list, axis=0)
        self.current_data_values = np.delete(self.indexed_data_values, self.del_list, axis=0)
        self.stack.append(self.current_data)
        self.stack_value.append(self.current_data_values)
        self.stack_index += 1 
        return 
    
    def go_back(self):
        if self.stack_index < 1:
            print("더 뒤로 갈 수 없습니다")
            return 
        self.stack_index -= 1 
        self.current_data = self.stack[self.stack_index]
        self.current_data_values = self.stack_value[self.stack_index]
        return 
    
    def go_forward(self):
        if self.stack_index >= len(self.stack_index)-1:
            print("더 앞으로로 갈 수 없습니다")
            return 
        self.stack_index += 1 
        self.current_data = self.stack[self.stack_index]
        self.current_data_values = self.stack_value[self.stack_index]
        return        
    