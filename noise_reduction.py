import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from Utility import *                                                                                                                                           

file_num_start = 328
file_num_finish = 400

for i in range(file_num_start, file_num_finish + 1):
    data = pd.read_csv('W2_data_{}.csv'.format(i))
    data_2 = data.values
    chopped_data = data_2[:, 16:18]

    del_list = noise_reduction(chopped_data)
    data = data.drop(del_list, axis=0)

    data.to_csv("./Done/W2_data_{}_.csv".format(i), index=None)
