from Utility import * 
import numpy as np 
import pandas as pd 

file_num_start = 326
file_num_finish = 326

for i in range(file_num_start, file_num_finish + 1):
    #Examplecode
    data = NoiseReducer()
    data.read(PATH='./Data/W2_data_{}.csv'.format(i))          #파일 불러오기
    data.show()                                                #데이터 그래프 그리기 
    data.chop(y_max=-1)                                        #데이터 자르기 
    data.show()
    data.noise_reduction()                                     #역행 없애기 (완벽하진 않음)
    data.show()
    data.go_back()
    data.show()
    data.save(PATH='./Test/W2_data_{}.csv'.format(i))

