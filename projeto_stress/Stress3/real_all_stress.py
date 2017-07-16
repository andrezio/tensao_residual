#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     16/07/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
from scipy.signal import savgol_filter

globaltwotheta=[]
globalpsi=[]

def multi():
    E=210000
    v=0.3
    theta2=156
    theta2/=2
    V=2.0*(1.0+v)
    theta = np.radians(theta2)
    theta = np.tan(theta)
    theta = 1.0/theta
    theta *= (np.pi/180.0)
    theta *=E
    theta /=-1.0*V
##    print theta #Mpq/deg
##    print theta/9.8#kg/mm2/deg
    return theta/9.8

def background(y):
    minimo=min(y)
    for i in range(len(y)):
        y[i]-=minimo
    return y

def normalizar(y):
    minimo=max(y)
    for i in range(len(y)):
        y[i]/=minimo
    return y

def check(file_name):
    global linha,psi
    vetor=['psi angle ' ,'<2Theta>']

    x=[]
    y=[]

    datafile = file(file_name)
    postion_intensity=False

    for line in datafile:

        if vetor[0] in line:
            linha= line.split()
            psi= (linha[3])
            psi=float(psi)

            globalpsi.append(psi)

        if postion_intensity:
            linha= line.split()
            valuex=(float(linha[0]))
            valuey=(float(linha[1]))
            x.append( valuex)
            y.append( valuey)

        if vetor[1] in line:
            postion_intensity=True

    x=np.asarray(x)
    y=np.asarray(y)

    y=background(y)
    y=savgol_filter(y,25,9)
    y=normalizar(y)
    plt.plot(x,y)
    mod = GaussianModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)

    twotheta=out.best_values['center']
    globaltwotheta.append(twotheta)


def getstress(file_name_old):
    check(file_name_old)



plt.figure(1)
plt.subplot(2,1,1)

file_name_fist = 'P_L_PB_1_//P_L_PB_1_.txt'
getstress(file_name_fist)

dados={'P_L_PB_1_':'P_L_PB_1_'}
files=range(1,11)
for i in files:
    data='%s%s//%s%s.txt'%( dados['P_L_PB_1_'],str(i),dados['P_L_PB_1_'],str(i))
    getstress(data)
plt.subplot(2,1,2)
plt.plot(globalpsi,globaltwotheta,'-o')
plt.show()