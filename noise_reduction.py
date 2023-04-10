from Utility import * 
import numpy as np 
import pandas as pd 

file_num_start = 326
file_num_finish = 400

for i in range(file_num_start, file_num_finish + 1):
    data = NoiseReducer()
    data.read(PATH='W2_data_{}.csv'.format(i))
    data.operate()
    data.save("./Done/W2_data_{}_.csv".format(i))
