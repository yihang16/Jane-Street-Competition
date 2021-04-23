import numpy as np 
import os  

variance = 0
filename = ''

for file in os.listdir('/home/yihang_toby/results'):
    if 'all' in file:
        var = np.loadtxt(file,dtype = np.float32)
        if var > variance:
            filename = file
            variance = var  
print(variance)
print(filename)