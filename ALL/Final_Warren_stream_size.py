import matplotlib.pyplot as plt
import scipy.signal
from math import *
##import funcoes as func
import pdb
import math
euler=2.718281828459045235360287
from numpy import loadtxt
from lmfit.models import GaussianModel, PolynomialModel, LinearModel,LognormalModel
from scipy.interpolate import *
from numpy import *
import numpy as np

x1,y1 = np.loadtxt('outsamplebalzar111.xy', unpack= True)
x11,y11 = np.loadtxt('outstandartbalzar111.xy', unpack= True)

x2,y2 = np.loadtxt('outsamplebalzar222.xy', unpack= True)
x22,y22 = np.loadtxt('outstandartbalzar222.xy', unpack= True)


def removerbackground(y):
    maximo=max(y)
    for i in range(len(y)):
        y[i]=(y[i]/maximo)

    minimo=min(y)
    for i in range(len(y)):
        y[i]-=minimo

    return y

removerbackground(y1)
removerbackground(y11)
removerbackground(y2)
removerbackground(y22)

def stokes(yn,ym,x1):
    yn=np.fft.rfft(yn)
    ym=np.fft.rfft(ym)
    newvetor=[]
    newL=[]
    lambida=0.154
    theta1=np.sin( np.radians( x1[0]/2))
    theta2=np.sin( np.radians( x1[-1]/2))

    baixo=2*(theta2-theta1)
    baixo=lambida/baixo

    for i in range(len(yn)):
        value=yn[i].real/ym[i].real
        if value<0:
            value=0


        newvetor.append(value)
        newL.append(i*baixo)

    for i in range(10,len(newvetor)):
        try:
            if newvetor[i]<newvetor[i+1]:
                try:
                    newvetor[i+1]=newvetor[i]
                except:
                    pass
        except:
            pass
    return newvetor,newL

y3,x3=stokes(y1,y11,x1)
y4,x4=stokes(y2,y22,x2)

manv=50
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(x3[:manv],y3[:manv],'-o')
p7= polyfit(x3[:manv],y3[:manv],9)
plt.plot(x3[:manv],polyval(p7,x3[:manv])[:manv],'-o',label='polinom: 9')
plt.subplot(1,2,2)
plt.plot(x4[:manv],y4[:manv],'-o')
p8= polyfit(x4[:manv],y4[:manv],9)
plt.plot(x4[:manv],polyval(p8,x4[:manv])[:manv],'-o',label='polinom: 9')
plt.show()

plt.figure(2)

vetorlen=range(2,100)
vetor1=[]
vetor3=[]
for i in vetorlen:
    vetor1.append(np.log(polyval(p7,i)))
    vetor3.append(np.log(polyval(p8,i)))



xx1=np.array([1]*len(vetor1))
xx3=np.array([8]*len(vetor3))

slope=[]
intercep=[]
mod = LinearModel()

for i in range(80):
    xxplot=[xx3[i],xx1[i]]
    yyplot=[vetor3[i],vetor1[i]]
    if i <=10:
        plt.plot(xxplot,yyplot,'-o')

    x=xxplot
    y=yyplot
    try:
        pars = mod.guess(y, x=x)
        out  = mod.fit(y, pars, x=x)

        if i==50:
            print 'microdeformacao: ', abs(out.values['slope']/(2*pi**2))

        slope.append(out.values['slope']/(2*pi**2))
        intercep.append(out.values['intercept'])
    except:
        pass


plt.show()

for i in range(len(intercep)):
    if intercep[i]<0:
        intercep[i]=0


plt.plot(intercep,'-o')
plt.show()

plt.plot(slope,'-o')
plt.show()