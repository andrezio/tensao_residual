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
from lmfit.models import GaussianModel,LinearModel
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
##    return theta/9.8#kg
    return theta#Mpa


def removekalpha(y,x):
    novoy=[]
    lambida2=1.541220
    lambida1=1.537400
    deltaL = lambida2 - lambida1
    deltaL = deltaL/lambida1
    diferenca=x[1]-x[0]

    for i in range(len(y)):
        deltasoma = x[1]-x[0]
        ase= np.tan(np.radians(x[i]/2))*2*deltaL/(diferenca)
        n=1;

        while(ase>deltasoma):
            deltasoma=deltasoma+diferenca
            n+=1
    	try:
    		yy=y[i]-0.5*y[i-n]

    		if yy<0:yy=(yy+y[i])/3

    		novoy.append(yy)
    	except:
    		novoy.append(y[i])
##    plt.plot(y)
##    normalizar(novoy)
##    plt.plot(novoy)
##    plt.show()

    return novoy

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
            psi=np.sin(np.radians(psi))**2

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
    y=savgol_filter(y,11,9)
    y=normalizar(y)
    y=removekalpha(y,x)
    plt.plot(x,y)
    mod = GaussianModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)

    twotheta=out.best_values['center']

    #center mass
##    twotheta=sum(y)/len(y)
    globaltwotheta.append(twotheta)


def getstress(file_name_old):
    check(file_name_old)


def lenar_calc(x,y):
    global dados
    mod = LinearModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)

##    print out.best_values
    plt.plot(x,out.best_fit)
    calc= out.best_values['slope']
##    print calc,multi()
    print 'Value:',dados, round( calc*multi(),3)

plt.figure(1)
plt.subplot(2,1,1)
plt.grid()

##file_name_fist = 'P_L_PB_1_//P_L_PB_1_.txt'
##getstress(file_name_fist)

##dados={'P_L_PB_2_':'P_L_PB_2_'}
dados='P_L_PB_7_'

files=range(1,11)
for i in files:
    data='%s%s//%s%s.txt'%( dados,str(i),dados,str(i))
    getstress(data)
plt.subplot(2,1,2)
plt.plot(globalpsi,globaltwotheta,'-o')

miny=int(min(globaltwotheta))-2
maxy=int(max(globaltwotheta))+2
maxx=round(max(globalpsi),3)+round(max(globalpsi),3)/2
plt.axis([0,maxx,miny,maxy])

plt.grid()
plt.xlabel('$\sin ^{2}\omega $')
plt.ylabel('$2\Theta $')

lenar_calc(globalpsi[:],globaltwotheta[:])
##plt.show()