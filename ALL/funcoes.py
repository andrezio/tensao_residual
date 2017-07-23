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
from math import *
from lmfit.models import LognormalModel
import pdb
from lmfit.models import  LinearModel
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import scipy.signal
from math import *

def normalizar(vetor):
    maximo=max(vetor)
    newvetor=[]
    for i in vetor:
        newvetor.append(i/maximo)

    return newvetor


def centralizar(y1,y2):
    def list(vetor):
        newvetor = []
        for i in vetor:
            newvetor.append(i)

        return newvetor

    y1=list(y1)
    y2=list(y2)
    pico1=y1.index(max(y1))
    pico2=y2.index(max(y2))

    n=pico1-pico2
    x=np.array([0]*abs(n))

    x1=np.array([0]*1000)

    if pico1<pico2:
        y1 =np.concatenate((x,y1))
    else:
        y2 =np.concatenate((x,y2))

    y22 =np.concatenate((x1,y2))
    y22 =np.concatenate((y22,x1))
    y11 =np.concatenate((x1,y1))
    y11 =np.concatenate((y11,x1))

    return y1,y2,y11,y22

def magnitude(FFT):
    for i in range(len(FFT)):
    	FFT[i]=sqrt(FFT[i].real**2+FFT[i].imag**2)
    return FFT


def removebackground(n,y,x):

    def list(vetor):
        newvetor = []
        for i in vetor:
            newvetor.append(i)

        return newvetor

    def minimo(y):
        minimo=min(y)
        for i in range(len(y)):
            y[i]-=minimo
        return y

    x1=list(x)
    y=list(y)
    Xn=[]


    y=minimo(y)#min values

    for i in x1[:n]:
        Xn.append(i)

    for i in x1[-n:]:
        Xn.append(i)

    mod = LinearModel()

    pars = mod.guess(y[:n]+y[-n:], x=Xn)
    out  = mod.fit(y[:n]+y[-n:], pars, x=Xn)

    m=out.values['slope']
    b=out.values['intercept']

    Z=m*x + b
    minimo = min(Z)
    for i in range(len(Z)):
        if Z[i]<minimo:
            Z[i]=minimo

    newy=y-Z

    newy=savgol_filter(newy, 15, 9)

    return newy

def harmonic_L(x,theta1,theta2,L):
    pass

def stokes(y1,y2):
    y1=y1.real
    y2=y2.real

    y=[]

    for i in range(len(y1)):
        try:
            if (y1[i]/y2[i])<=0:
                pass
            else:
                y.append((y1[i]/y2[i]))
        except:
            pass


    return y


def _getL(position,armonicos_stokes):
##    lambida=0.1033305
    lambida=0.154
    armonico=[]

    maior=0
    menor=0
    testemenor=True

    for i in range(len(position)):
        if not position[i]==0:

                menor = position[i]/2
                break

    for i in reversed( range(  len(position))):
        if not position[i]==0:

                maior = position[i]/2
                break

    baixo=((sin( radians( maior))-sin( radians( menor)))*2)

    for i in range(len(armonicos_stokes)):
        armonico.append(abs((i*lambida)/baixo) )

    return armonico


def _distribution(data):
    derivat=(np.gradient(data))
    derivat = np.gradient(derivat)
    for i in range(len(derivat)):
        if derivat[i]<0:
            derivat[i]=0
    return derivat