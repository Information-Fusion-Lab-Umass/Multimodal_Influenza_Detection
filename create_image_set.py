import os
import numpy as np

with open('osu.txt','w') as f:
	for i in range(284):
		f.write('I'+'%05d'%i)
		f.write('\n')
