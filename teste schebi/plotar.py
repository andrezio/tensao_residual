#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lgallego
#
# Created:     08/08/2016
# Copyright:   (c) lgallego 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

#https://docs.scipy.org/doc/numpy/reference/routines.polynomials.chebyshev.html

namefile="Riella_P_Fe_1.xy"

def normalizar(y):
	maximo=max(y)
	for i in range(len(y)):
		y[i]/=maximo
	return y

x,y = np.loadtxt(namefile, unpack= True)
normalizar(y)
y2=np.polynomial.chebyshev.chebfit(x,y,6,rcond=0.0, full=False, w=None)
print y2
plt.plot(x,y,'red')

y3= np.polynomial.chebyshev.chebval(x,y2)
#normalizar(y3)
plt.plot(x,y3)
#plt.plot(x,y-y3)

plt.show()
