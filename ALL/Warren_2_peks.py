import matplotlib.pyplot as plt
import scipy.signal
from math import *
import funcoes as func
import pdb
import math
euler=2.718281828459045235360287
from numpy import loadtxt
from lmfit.models import GaussianModel, PolynomialModel, LinearModel,LognormalModel
from scipy.interpolate import *
from numpy import *
import numpy as np

dicionariop={}

n=1
L=2
C=3

def plot_fft(y1,x1,y11,x11,name,titulo):
    global n,L,C
    name=name.split('.')[0]
    plt.suptitle(titulo, fontsize=16)

    y1=func.normalizar(y1)
    y11=func.normalizar(y11)


    y1=func.removebackground(4,y1,x1)
    y11=func.removebackground(4,y11,x11)

    y1=func.normalizar(y1)
    y11=func.normalizar(y11)

    plt.figure(1)

    plt.subplot(L,C,n)
    n+=1
    plt.plot(x1,y1,'r-o',label='sample')
    plt.plot(x11,y11,'-o',label='std')
    plt.xlabel('2Theta')
    plt.ylabel('Intensity')
    plt.title('Plot')
    plt.legend()

    y1,y11,y1x,y11x=func.centralizar(y1,y11)

    y1fft=np.fft.rfft(y1x)
    y11fft=np.fft.rfft(y11x)


    plt.subplot(L,C,n)
    plt.title('Stokes Deconvolution')
    n+=1

    stokes=func.stokes(y1fft,y11fft)
    m=100
    armonico = func._getL(x1,stokes)
    pstokes=stokes[0]
    for i in range(30,len(stokes)):
        if stokes[i]>pstokes:
            m=i
            break


    plt.plot(armonico[:m],stokes[:m],'-o',label='deconvolution')

    yp = stokes[:m]
    xp = armonico[:m]

    p7= polyfit(xp,yp,9)
    dicionariop[name]=p7
    plt.plot(xp,polyval(p7,xp),label='polinom: 9')

    plt.xlabel('n(nm)')
    plt.ylabel('A(n)')
    plt.legend()



    plt.subplot(L,C,n)
    plt.title('Distribuition Cristalite Size')
    n+=1
    plt.plot( armonico[:m],func._distribution( stokes[:m]),'-o',label='distribuition')
##    mod=LognormalModel
##    x=armonico[:m]
##    pars = mod.guess(y, x=x)
##    out  = mod.fit(y, pars, x=x)
    plt.xlabel('n(nm)')
    plt.ylabel('A(n)')
    plt.legend()



x1,y1 = np.loadtxt('outsamplebalzar111.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartbalzar111.xy', unpack= True)
plot_fft(y1,x1,y11,x11,'balzar111.png','Balzar')
x1,y1 = np.loadtxt('outsamplebalzar222.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartbalzar222.xy', unpack= True)
plot_fft(y1,x1,y11,x11,'balzar222.png','Balzar')

plt.show()