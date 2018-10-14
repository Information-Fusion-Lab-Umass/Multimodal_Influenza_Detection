import os
import numpy as np

with open('imagesetfile.txt','w') as f:
	for i in range(2020):
		f.write('I'+'%05d'%i)
		f.write('\n')
