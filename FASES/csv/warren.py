import matplotlib.pyplot as plt
import scipy.signal
from math import *
import pdb
import math
from numpy import loadtxt
from lmfit.models import GaussianModel, PolynomialModel, LinearModel,LognormalModel
from scipy.interpolate import *
from numpy import *
import numpy as np


lambida=0.154
def normalizar(y):
    maximo=max(y)
    for i in range(len(y)):
        y[i]=y[i]/maximo
    return y
def warren(x1,y1,x2,y2):
    #LINE
    L=2
    #COLLUM
    C=4
    #IMAGE
    P=1
    def stokes(yn,x1):
        yn=np.fft.rfft(yn)


        for i in range(len(yn)):
            yn[i]=np.sqrt(yn[i].real**2+yn[i].imag**2)


        newvetor=[]
        newL=[]

        theta1=np.sin( np.radians( x1[0]/2))
        theta2=np.sin( np.radians( x1[-1]/2))

        baixo=2*(theta2-theta1)
        baixo=lambida/baixo

        for i in range(len(yn)):
            value=yn[i].real
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

    y3,x3=stokes(y1,x1)
    y4,x4=stokes(y2,x2)


    plt.figure(1)
    plt.subplot(L,C,P);P=P+1
    plt.plot(x3,y3)
    plt.subplot(L,C,P);P=P+1
    plt.plot(x4,y4)
    manv=50
##    plt.figure(1)
    plt.subplot(L,C,P);P=P+1
    plt.xlabel('$L(nm)$')
    plt.ylabel("$L_A(nm)$")
    plt.plot(x3[:manv],y3[:manv],'-o')
    p7= polyfit(x3[:manv],y3[:manv],9)
    plt.plot(x3[:manv],polyval(p7,x3[:manv])[:manv],'-o',label='111')
    plt.legend()
    plt.subplot(L,C,P);P=P+1
    plt.plot(x4[:manv],y4[:manv],'-o')
    p8= polyfit(x4[:manv],y4[:manv],9)
    plt.plot(x4[:manv],polyval(p8,x4[:manv])[:manv],'-o',label='222')
    plt.xlabel('$L(nm)$')
    plt.ylabel("$L_A(nm)$")
    plt.legend()
##    plt.show()



    vetorlen=range(2,100)
    vetor1=[]
    vetor3=[]
    for i in vetorlen:
        vetor1.append(np.log(polyval(p7,i)))
        vetor3.append(np.log(polyval(p8,i)))


    d1=28.724
    d2=59.445

    d1=lambida/(2*np.sin ( np.radians( d1/2 )))
    d2=lambida/(2*np.sin ( np.radians( d2/2 )))

    xx1=np.array([1/d1**2]*len(vetor1))
    xx3=np.array([1/d2**2]*len(vetor3))

    slope=[]
    intercep=[]
    mod = LinearModel()


    plt.subplot(L,C,P);P=P+1
    for i in range(80):
        xxplot=[xx3[i],xx1[i]]
        yyplot=[vetor1[i],vetor3[i]]
        if  i<30:
            plt.plot(xxplot,yyplot,'-o')

        x=xxplot
        y=yyplot
        try:
            pars = mod.guess(y, x=x)
            out  = mod.fit(y, pars, x=x)

            if i==50:
                print 'microdeformacao: ', abs(out.values['slope']/(2*pi**2))

            slope.append(  ( out.values['slope']/(2*pi**2)))
            intercep.append( np.e*( out.values['intercept'] ) )
        except:
            pass

    plt.xlabel('$1/d^2(nm^2)$')
    plt.ylabel("$LN(L_A(nm))$")
##    plt.show()


    for i in range(len(intercep)):
        intercep[i]=np.e**intercep[i]
        if intercep[i]<0:
            intercep[i]=0
        if slope[i]<0:
            slope[i]=0



    plt.subplot(L,C,P);P=P+1
    plt.plot(intercep,'-o')
    plt.xlabel('$L(nm)$')
    plt.ylabel("$L_v(nm)$")
##    plt.show()

    derivada=np.gradient(np.gradient(intercep))
    for i in range(len(derivada)):

        if derivada[i]<=0:
            derivada[i]=0

    derivada=normalizar(derivada)
    np.savetxt('test.out', derivada, delimiter=',')
    plt.subplot(L,C,P);P=P+1
    plt.plot(derivada,'-o')
    plt.title('distuibution')

    plt.xlabel('$L(nm)$')
    plt.ylabel("$Distribution$")
##    plt.show()
    plt.subplot(L,C,P);P=P+1
    plt.xlabel('$L(nm)$')
    plt.ylabel("$RMSS$")
    plt.plot(slope,'-o')
    plt.show()






