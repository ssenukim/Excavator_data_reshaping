import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import copy

class NoiseReducer():
    def __init__(self):
        self.data = None 
        self.current_data = None
        self.current_data_values = None
        self.data_values = None
        self.indexed_data_values = None 
        self.angle_values = None 
        self.time_data = None 
        self.del_list = []
        self.del_list_stack = []
        self.stack = []
        self.stack_value = []
        self.stack_index = -1

    def read(self, PATH):
        self.data = pd.read_csv(PATH)
        self.current_data = self.data 
        self.data_values = self.data.values[:, 16:]
        #self.angle_values = self.data.values[:, 6:9]
        self.time_data_str = self.data.values[:, 14]
        self.time_data = self.time_data_str.reshape(len(self.time_data_str), 1)
        self.current_data_values = np.concatenate((self.data_values, self.time_data), axis=1)
        
        def indexing(data):
            index_array = np.arange(0, data.shape[0] , 1, dtype=int).T
            index_array = index_array.reshape((len(index_array), 1)) 
            result_data = np.concatenate((data, index_array), axis=1)               
            return result_data
        
        self.current_data_values = indexing(self.current_data_values)
        self.indexed_data_values = self.current_data_values
        self.stack.append(self.current_data)
        self.stack_value.append(self.current_data_values)
        self.stack_index += 1 
        print("read finished")
        return 
    
    def save(self, PATH):
        self.current_data.to_csv(PATH, index=None)
        return 
       
    def show(self, show_AOA= True, bar_length=0.35):
        x_val = self.current_data_values[:, 0].squeeze().astype('float64')
        y_val = self.current_data_values[:, 1].squeeze().astype('float64')
        aoa = self.current_data_values[:, 2].squeeze().astype('float64')
        aoa = ((180 + aoa)/180)*np.pi
        function = interpolate.interp1d(x_val, y_val, kind='linear')
        x_new = np.linspace(np.min(x_val), np.max(x_val), num=100, endpoint=True, dtype=float)
        try:
            y_new = function(x_new)
            fig, ax = plt.subplots()
            ax.plot(x_val, y_val, 'o', x_new, y_new, '-')
            plt.legend(['Point', 'linear'])
            if show_AOA==True:
                for i in range(len(x_val)):
                    x_1 = x_val[i] + (bar_length/2)*math.cos(aoa[i])
                    y_1 = y_val[i] + (bar_length/2)*math.sin(aoa[i])
                    x_2 = x_val[i] - (bar_length/2)*math.cos(aoa[i])
                    y_2 = y_val[i] - (bar_length/2)*math.sin(aoa[i])
                    ax.plot([x_1, x_2], [y_1, y_2], linestyle='--', linewidth=2, color='r')

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
                self.del_list.append(self.current_data_values[i,-1])
                continue 
            
            if self.current_data_values[i, 1] < y_min or self.current_data_values[i, 1] > y_max:
                self.del_list.append(self.current_data_values[i,-1])
        
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
                self.del_list.append(self.current_data_values[i, -1])
                continue 

            if i<len(self.current_data_values) - 3 and self.current_data_values[i+1, 0] - self.current_data_values[i, 0] < 0 and self.current_data_values[i+2, 0] - self.current_data_values[i+1, 0] > 0 and self.current_data_values[i+3, 0] - self.current_data_values[i+2, 0] <0:
                self.del_list.append(self.current_data_values[i+1, -1])
                self.del_list.append(self.current_data_values[i+2, -1])

        self.del_list = list(set(self.del_list))
        self.current_data = self.data.drop(self.del_list, axis=0)
        self.current_data_values = np.delete(self.indexed_data_values, self.del_list, axis=0)
        self.stack.append(self.current_data)
        self.stack_value.append(self.current_data_values)
        self.stack_index += 1 
        return 
    
    def noise_reduction_2(self, param_1=75, param_2=50):
        i = 0
        while i > self.current_data_values.shape[0] - 4:
            vec_1 = (self.current_data_values[i+1, :2] - self.current_data_values[i, :2]).squeeze()
            vec_2 = (self.current_data_values[i+2, :2] - self.current_data_values[i+1, :2]).squeeze()
            if np.dot(vec_1, vec_2) <0: 
                self.del_list.append(self.current_data_values[i+1, -1])
                if i > self.current_data_values.shape[0] -3:
                    break
                vec_3 = (self.current_data_values[i+3, :2] - self.current_data_values[i+2, :2]).squeeze()
                def get_angle(vector_1, vector_2):
                    abs_vec_1 = np.linalg.norm(vector_1)
                    abs_vec_2 = np.linalg.norm(vector_2)
                    dot = np.dot(abs_vec_1, abs_vec_2)
                    angle = math.acos((dot)/(abs_vec_1*abs_vec_2))
                    return math.degrees(angle)
                if get_angle(vec_1, vec_3) > param_1:
                    vec_4 = (self.current_data_values[i+2, :2] - self.current_data_values[i, :2]).squeeze()
                    if get_angle(vec_3, vec_4) > param_2:
                        self.del_list.append(self.current_data_values[i+2, -1])
                        i += 3 
                        continue 
                    else: 
                        i += 2 
                        continue
                i += 2
                continue 
            i += 1 
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
    
    def sample_point(self, period=0.01):
        time_data = self.current_data_values[:, 3]
        def time_str2float(data):
            result_data = np.zeros(len(data))
            for i in range(len(data)):
                str = data[i]
                sec = str[-6:]
                min = str[-8: -6]
                sec = float(sec)
                min = float(min)
                result_data[i] = sec + 60 * min
            return result_data
        time_data_sec = time_str2float(time_data) 
        new_time_data = np.arange(time_data_sec[0], time_data_sec[-1], period)
        new_data = np.zeros((len(new_time_data), self.current_data.values.shape[1]))
        current_data_values = self.current_data.values
        for i in range(current_data_values.shape[1]):
            if i==14 or i==15:    
                continue
            print(current_data_values.shape, type(current_data_values), i)
            y_val = self.current_data.values[:, i].astype('float64')
            function = interpolate.interp1d(time_data_sec, y_val, kind='linear')
            y_new = function(new_time_data)
            new_data[:, i] = y_new

        print(new_data.shape)
        for i in range(new_data.shape[0]):
            minute = str(int(new_time_data[i]//60))
            second = str(round(new_time_data[i]%60, 3))
            string = self.current_data.values[1, 14]
            string = string[:-8]
            sum = string + minute + second 
            new_data[i, 14] = sum
            string_2 = self.current_data.values[0, 15]
        
        return new_data
        
    def remove_overlap(self):
        for i in range(self.current_data_values.shape[0]-1):
            if abs(self.current_data_values[i, 0] - self.current_data_values[i+1, 0]) < 0.001:
                self.del_list.append(self.current_data_values[i+1, -1])

        self.del_list = list(set(self.del_list))
        self.current_data = self.data.drop(self.del_list, axis=0)
        self.current_data_values = np.delete(self.indexed_data_values, self.del_list, axis=0)
        self.stack.append(self.current_data)
        self.stack_value.append(self.current_data_values)
        self.stack_index += 1        
        return      

