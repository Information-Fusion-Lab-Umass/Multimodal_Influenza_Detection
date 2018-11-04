import os
import numpy as np

with open('imagesetfile_sample_test.txt','w') as f:
	for i in range(1000):
		f.write('I'+'%05d'%i)
		f.write('\n')
