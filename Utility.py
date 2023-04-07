import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def noise_reduction(data:np.ndarray) -> np.ndarray:
    del_list = []
    for i in range(len(data)):
        if data[i, 1] >= 0:
            del_list.append(i)
            continue 

        if data[i, 0] - data[i-1, 0] > 0: 
            if i==0:
                continue 
            del_list.append(i-1)
            continue     

    return del_list
