#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     11/06/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from math import *
import funcoes as func

def plot_fft(y1,x1,y11,x11,name):
    name=name.split('.')[0]
    plt.suptitle(name, fontsize=16)

    y1=func.normalizar(y1)
    y11=func.normalizar(y11)


    y1=func.removebackground(4,y1,x1)
    y11=func.removebackground(4,y11,x11)

    y1,y11,y1x,y11x=func.centralizar(y1,y11)


    plt.figure(1)
    #LInha de cima
    plt.subplot(1,2,1)

    plt.plot(y1,'r-o',label='sample')
    plt.plot(y11,'-o',label='std')
    plt.legend()

    y1fft=np.fft.rfft(y1x)
    y11fft=np.fft.rfft(y11x)

    plt.subplot(1,2,2)
    m=100
    plt.plot(x1[:m],func.stokes(y1fft,y11fft)[:m],'-o',label='stokes')
    plt.legend()

##    plt.close()
    plt.show()

x1,y1 = np.loadtxt('outsampleky03111.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartky03111.xy', unpack= True)
plot_fft(y1,x1,y11,x11,'ky3111.png')

x1,y1 = np.loadtxt('outsampleky03222.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartky03222.xy', unpack= True)
##plot_fft(y1,y11,'ky3222.png')
x1,y1 = np.loadtxt('outsampleky03444.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartky03444.xy', unpack= True)
##plot_fft(y1,y11,'ky3444.png')
x1,y1 = np.loadtxt('outsampleuo2_222.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartuo2_222.xy', unpack= True)
##plot_fft(y1,y11,'uo2222.png')
x1,y1 = np.loadtxt('outsampleuo2_111.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartuo2_111.xy', unpack= True)
##plot_fft(y1,y11,'uo2111.png')
x1,y1 = np.loadtxt('outsamplebalzar111.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartbalzar111.xy', unpack= True)
##plot_fft(y1,y11,'balzar111.png')
x1,y1 = np.loadtxt('outsamplebalzar222.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartbalzar222.xy', unpack= True)
##plot_fft(y1,y11,'balzar222.png')


###linha de baixo
##plt.subplot(2,3,4)
##
##plt.plot(y1,'r-o',label='sample')
##plt.plot(y11,'-o',label='std')
##plt.legend()
##
##plt.subplot(2,3,5)
##
##y1fft=np.fft.rfft(y1)
##y11fft=np.fft.rfft(y11)
##
##plt.plot(y1fft,'r-o',label='sample')
##plt.plot(y11fft,'-o',label='std')
##plt.legend()
##
##plt.subplot(2,3,6)
##
##plt.plot( func.magnitude( scipy.signal.deconvolve(y1fft,y11fft)[1]  ),'r-o',label='scipy')
##plt.plot( func.magnitude(np.fft.rfft(y1))  )
##plt.legend()


