#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andre
#
# Created:     03/07/2017
# Copyright:   (c) Andre 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from lmfit.models import VoigtModel, GaussianModel,LinearModel
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
import pdb
from scipy import stats


mintheta=0
maxtheta=0




def lambida_func():
    #lambida=0.1033305
    lambida=0.154
    return lambida

#Func
def normalizar(vetor):
    maximo=max(vetor)
    newvetor=[]
    for i in vetor:
        newvetor.append(i/maximo)
    return newvetor

def stokes(y,y1):
    yy=[]
    for i in range(len(y)):
        try:
            yy.append(y[i].real/y1[i].real)
        except:
            pass
    return yy

def centralizarPlot(y1,y2):
    def list(vetor):
        newvetor = []
        for i in vetor:
            newvetor.append(i)

        return newvetor

    y1=normalizar(y1)
    y2=normalizar(y2)

    y1=list(y1)
    y2=list(y2)
    pico1=y1.index(max(y1))
    pico2=y2.index(max(y2))

    n=pico1-pico2
    x=np.array([0]*abs(n))

    if pico1<pico2:
        y1 =np.concatenate((x,y1))
    else:
        y2 =np.concatenate((x,y2))

    return y1,y2

def addpoint(y1,y2):
    x1=np.array([0]*1000)
    y22=[]
    y11=[]

    y22 =np.concatenate((x1,y2))
    y22 =np.concatenate((y22,x1))
    y11 =np.concatenate((x1,y1))
    y11 =np.concatenate((y11,x1))

    return y11,y22

def removebackground(y,x,n=4):
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

    #newy=savgol_filter(newy, 15, 9)

    return newy

def magnitude(FFT):
    for i in range(len(FFT)):
    	FFT[i]=np.sqrt(FFT[i].real**2+FFT[i].imag**2)
    return FFT



def savitzgolay_filter(y,w=51,p=9):
    return savgol_filter(y,w,p)

def removerBackground(y,n=8):
    minimo = min(y)
    for i in range(len(y)):
        y[i]-=minimo
        if i <=n or i>=(len(y)-n):
            y[i]=0.0
    return y

def Decon_Lor(s1,s2):
    s1=np.radians(s1)
    s2=np.radians(s2)

    return (s1-s2)

def Decon_Gau(s1,s2):
    s1=np.radians(s1)
    s2=np.radians(s2)

    return (np.sqrt(pow(s1,2)-pow(s2,2)))


#Gauss
def ScherrerEquation(sigma,center):
    lambida=lambida_func()
##    print 'Sigma:', sigma
    center=np.cos(np.radians(center/2))
    D=(0.9*lambida)/(2.3548*sigma*center)
    print 'D:', int(D), 'nm'

def GaussCalc(x,y,x1,y1):
    y=removerBackground(y)
    y1=removerBackground(y1)

    mod = GaussianModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)


    mod = GaussianModel()
    pars1 = mod.guess(y1, x=x1)
    out1  = mod.fit(y1, pars1, x=x1)

    center=out.best_values['center']
    sigma = Decon_Gau(out.best_values['sigma'],out1.best_values['sigma'])
    return ScherrerEquation(sigma, center)


#Single line
def SingleLineEquation(sigma, gamma, center):
    lambida=lambida_func()
    tancenter=np.tan(np.radians(center/2))

##    print 'Sigma:', sigma
##    print 'Gamma:', gamma

    center=np.cos(np.radians(center/2))

    D=(1.0*lambida)/(2.0*gamma*(np.pi/2.0)*center)
    e=2.3548*sigma*(1.06446701943)/(tancenter*4.0)

##    e=e**2
    print 'D:', int(D), ' nm'
    print '<e>:',round(e,3)

def VoigtCalc(x,y,x1,y1):
    y=removerBackground(y)
    y1=removerBackground(y1)

    mod = VoigtModel()
    pars = mod.guess(y, x=x)
    pars['gamma'].set(value=0.8, vary=True, expr='')
    out  = mod.fit(y, pars, x=x)

    mod = VoigtModel()
    pars1 = mod.guess(y1, x=x1)
    pars1['gamma'].set(value=0.8, vary=True, expr='')
    out1  = mod.fit(y1, pars1, x=x1)


##    print out.best_values
##    print out1.best_values

    center=out.best_values['center']
    sigma = Decon_Gau(out.best_values['sigma'],out1.best_values['sigma'])
    gamma = Decon_Lor(out.best_values['gamma'],out1.best_values['gamma'])
    return SingleLineEquation(sigma, gamma, center)


def LinearWarren(x,y):
    global mintheta, maxtheta

    mod = LinearModel()

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    XS=out.values['intercept']/out.values['slope']*-1
    XS=int(XS)
    La=XS
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
##    print 'SLOPE ',slope
##    print 'Intercepe ', intercept
    Da = lambida_func()*intercept/(2*(np.sin(np.radians( maxtheta/2))-np.sin(np.radians( mintheta/2)) ))
    print 'D',round(Da,2), 'nm'

    # Create a list of values in the best fit line
    abline_values = [slope * i + intercept for i in x]

    lx=[]
    ly=[]
    endwhile=True
    i=0
    while(endwhile):
        lx.append(i)
        valuey=slope * i + intercept
        ly.append(valuey)
        if valuey<=0:
            endwhile=False
        else:
            i+=1
            valuey=0

    #pdb.set_trace()

    plt.plot(lx,ly, 'red')

#Warren
def warren_averbac(x,y,x1,y1,m=20):
    global mintheta, maxtheta
    mintheta=x[0]
    maxtheta=x[-1]
    y=normalizar(y)
    y1=normalizar(y1)
    y,y1=centralizarPlot(y,y1)
##    y,y1=addpoint(y,y1)
    y=np.fft.rfft(y)
    y1=np.fft.rfft(y1)
    y=magnitude(y)
    y1=magnitude(y1)
    yy=stokes(y,y1)
    #pdb.set_trace()
    plt.subplot(1,2,2)
    plt.plot(yy[:m],'-o')
    warreny=yy[0:6]
    #pdb.set_trace()
    yy=LinearWarren(range(len(warreny)),warreny)

    #plt.plot(yy[:m],'-o')
    plt.show()


def Plotar(x,y,x1,y1):
    y=removerBackground(y)
    y1=removerBackground(y1)

##    y,y1=centralizarPlot(y,y1)

    plt.plot(x,y)
    plt.plot(x1,y1)
