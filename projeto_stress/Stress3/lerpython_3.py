from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel,PseudoVoigtModel, LinearModel,GaussianModel
from scipy.signal import savgol_filter
from Tkinter import *
from ttk import *
import math
import numpy as np

E=210000
v=0.3

#Mudar o nome do arquivo
file_name_old = 'P_L_PB_1_//P_L_PB_1_.txt'

def normalizar(y):
    minimo=max(y)
    for i in range(len(y)):
        y[i]/=minimo
    return y


def removerkalhpa2(y,x):
    lambida2=1.541220
    lambida1=1.537400

    deltaL = lambida2 - lambida1
    deltaL = deltaL/lambida1

    diferenca=x[1]-x[0]

    m=x[1]-x[0]

    minimo=min(y)

    novoy=[]

    for i in range(len(y)):
        deltasoma = x[1]-x[0]
        ase= math.tan(math.radians(x[i]/2))*2*deltaL/(diferenca)
        n=0

        while(ase>deltasoma):
            deltasoma=deltasoma+diferenca
            n+=1

        n+=1
        print n
        try:
            yy=y[i]-0.5*y[i-n]
            novoy.append(yy)
        except:
            novoy.append(y[i])
        minimo=min(y)

        for i in range(len(novoy)):

            if novoy[i]<0:
                novoy[i]=(novoy[i]+y[i])/3

    return novoy


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
            #print linha
            psi= (linha[3])
            psi=float(psi)

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

    return x,y

def background(y):
    minimo=min(y)
    for i in range(len(y)):
        y[i]-=minimo
    return y

def getstress(file_name_old):


    x,y=check(file_name_old)


    y=background(y)
    y=savgol_filter(y,25,9)
    y=normalizar(y)

    plt.plot(x,y,label='data')




    y1=removerkalhpa2(y,x)
    y1=savgol_filter(y1,25,9)
    y1=normalizar(y1)
    plt.plot(x,y1,label='nokalphaa2')
    plt.legend()
    plt.grid()
    plt.show()



getstress(file_name_old)
