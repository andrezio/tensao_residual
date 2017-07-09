#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andre
#
# Created:     04/07/2017
# Copyright:   (c) Andre 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gauss as functions


x,y = np.loadtxt('outsampleky03111.xy', unpack= True)
x1,y1 = np.loadtxt('outstandartky03111.xy', unpack= True)

x,y = np.loadtxt('outsample111.xy', unpack= True)
x1,y1 = np.loadtxt('outstandart111.xy', unpack= True)

x,y = np.loadtxt('outsample222.xy', unpack= True)
x1,y1 = np.loadtxt('outstandart222.xy', unpack= True)


#Plot
plt.figure(1)
plt.subplot(1,2,1)
functions.Plotar(x,y,x1,y1)

print '### Scherrer ###'
functions.GaussCalc(x,y,x1,y1)

print '### Single Line ###'
functions.VoigtCalc(x,y,x1,y1)

print "### WARREN ###"
functions.warren_averbac(x,y,x1,y1)
