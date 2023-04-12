import os 

start = 373
end = 400
Weird_data_index_list = [372, 379]
for i in range(start, end+1):
    os.rename('./W2_data_{}_.csv'.format(i),'./W2_data_{}.csv'.format(i) )

#372 
#379  