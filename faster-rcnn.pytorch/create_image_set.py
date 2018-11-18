import os
import numpy as np

with open('imagesetfile_overfit_test.txt','w') as f:
	for i in range(330,335):
		f.write('I'+'%05d'%i)
		f.write('\n')
