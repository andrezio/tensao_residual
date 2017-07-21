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
import pdb
import math
euler=2.718281828459045235360287

from numpy import loadtxt
from lmfit.models import GaussianModel, PolynomialModel, LinearModel
from scipy.interpolate import *
from numpy import *
import numpy as np
import matplotlib.pyplot as plt


dicionariop={}

n=1
L=3
C=3


def plot_fft(y1,x1,y11,x11,name):
    global n,L,C
    name=name.split('.')[0]
    plt.suptitle('Ky30', fontsize=16)

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
    plt.legend()

    y1,y11,y1x,y11x=func.centralizar(y1,y11)

    y1fft=np.fft.rfft(y1x)
    y11fft=np.fft.rfft(y11x)


    plt.subplot(L,C,n)
    n+=1

    stokes=func.stokes(y1fft,y11fft)

    armonico = func._getL(x1,stokes)

    m=95


    plt.plot(armonico[:m],stokes[:m],'-o',label='stokes')

    yp = stokes[:m]
    xp = armonico[:m]

    p7= polyfit(xp,yp,9)
    dicionariop[name]=p7
    plt.plot(xp,polyval(p7,xp),label='polinom: 9')

    plt.xlabel('n(nm)')
    plt.ylabel('A(n)')
    plt.legend()

    m=300

    plt.subplot(L,C,n)
    n+=1
    plt.plot( armonico[:m],func._distribution( stokes[:m]),'-o',label='stokes')
    plt.xlabel('n(nm)')
    plt.ylabel('A(n)')
    plt.legend()



##x1,y1 = np.loadtxt('outsampleuo2_222.xy', unpack= True)
##x11,y11 = np.loadtxt('outstandartuo2_222.xy', unpack= True)
##plot_fft(y1,x1,y11,x11,'ky3222.png')
##x1,y1 = np.loadtxt('outsampleuo2_111.xy', unpack= True)
##x11,y11 = np.loadtxt('outstandartuo2_111.xy', unpack= True)
##plot_fft(y1,x1,y11,x11,'ky3444.png')

x1,y1 = np.loadtxt('outsampleky03111.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartky03111.xy', unpack= True)
plot_fft(y1,x1,y11,x11,'ky3111.png')
##x1,y1 = np.loadtxt('outsampleky03222.xy', unpack= True)
##x11,y11 = np.loadtxt('outstandartky03222.xy', unpack= True)
##plot_fft(y1,x1,y11,x11,'ky3222.png')
x1,y1 = np.loadtxt('outsampleky03444.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartky03444.xy', unpack= True)
plot_fft(y1,x1,y11,x11,'ky3444.png')


##x1,y1 = np.loadtxt('outsamplebalzar111.xy', unpack= True)
##x11,y11 = np.loadtxt('outstandartbalzar111.xy', unpack= True)
##plot_fft(y1,x1,y11,x11,'ky3111.png')
##x1,y1 = np.loadtxt('outsamplebalzar222.xy', unpack= True)
##x11,y11 = np.loadtxt('outstandartbalzar222.xy', unpack= True)
##plot_fft(y1,x1,y11,x11,'ky3222.png')



plt.show()

vetor1=[]
vetor2=[]
vetor3=[]

for i in range(150):
    vetor1.append(math.log(polyval(dicionariop['ky3111'],i)))
##    vetor2.append(math.log(polyval(dicionariop['ky3222'],i)))
    vetor3.append(math.log(polyval(dicionariop['ky3444'],i)))


xx1=np.array([1]*len(vetor1))
##xx2=np.array([4]*len(vetor2))
xx3=np.array([8]*len(vetor3))

mod = LinearModel()

slope=[]
intercep=[]
xpp=[]
plt.figure(2)
plt.subplot(2,2,1)
plt.xlabel('n')
plt.ylabel('Ln(An)')

plotagem=[5,10,15,25,40,50]

for i in range(150):


    xplot=[xx1[i],xx3[i]]
    yplot=[euler**vetor1[i],euler**vetor3[i]]
    if  i in plotagem:
        plt.plot(xplot,yplot,'-o')

##    xplot=[xx1[i],xx2[i],xx3[i]]
##    yplot=[euler**vetor1[i],euler**vetor2[i],euler**vetor3[i]]
##    if not i >15:
##        plt.plot(xplot,yplot,'-o')

##    xplot=[xx2[i],xx3[i]]
##    yplot=[euler**vetor2[i],euler**vetor3[i]]
##    if not i >15:
##        plt.plot(xplot,yplot,'-o')

##    xplot=[xx1[i],xx3[i]]
##    yplot=[euler**vetor1[i],euler**vetor3[i]]
##    plt.plot(xplot,yplot,'-o')

    x=xplot
    y=yplot

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)

    if i==50:
        print 'microdeformacao: ', abs(out.values['slope']/(2*pi**2))

    slope.append(out.values['slope']/(2*pi**2))
    intercep.append(out.values['intercept'])

    xpp.append(i)



plt.subplot(2,2,2)
plt.xlabel('n')
plt.ylabel('<e>')
plt.plot(slope,'-o')
plt.subplot(2,2,3)
plt.xlabel('n(nm)')
plt.ylabel('(An)')
plt.plot(intercep,'-o')


yp = intercep
xp = xpp
p7= polyfit(xp,yp,9)
plt.plot(polyval(p7,xp),label='polinom: 9')



plt.subplot(2,2,4)
plt.xlabel('n')
plt.ylabel('An')
derivada = func._distribution(yp)
plt.plot(xp,derivada,'-o')

plt.show()