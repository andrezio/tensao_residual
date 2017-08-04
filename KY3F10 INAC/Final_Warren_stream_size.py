import matplotlib.pyplot as plt
import scipy.signal
from math import *
import pdb
import math
from numpy import loadtxt
from lmfit.models import GaussianModel, PolynomialModel, LinearModel,LognormalModel,VoigtModel
from scipy.interpolate import *
from numpy import *
import numpy as np
lambida=0.1033305
from scipy.signal import savgol_filter
from scipy import stats

#LINE
L=2
#COLLUM
C=4
#IMAGE
P=1


##x1,y1 = np.loadtxt('outsample002.xy', unpack= True)
##x11,y11 = np.loadtxt('outstandart002.xy', unpack= True)
##
##x2,y2 = np.loadtxt('outsample004.xy', unpack= True)
##x22,y22 = np.loadtxt('outstandart004.xy', unpack= True)

x1,y1 = np.loadtxt('outsample101zno70.xy', unpack= True)
x11,y11 = np.loadtxt('outstandart101zno70.xy', unpack= True)

x2,y2 = np.loadtxt('outsample202zno70.xy', unpack= True)
x22,y22 = np.loadtxt('outstandart202zno70.xy', unpack= True)



def getcenter(x,y):
    mod=VoigtModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    return out.best_values['center']

def interceptarwarren(y):
    x=range(len(y))
    mod = LinearModel()


    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:6],y[:6])
    abline_values = [slope * i + intercept for i in x]

    abline_values2=[]

    for i in abline_values:
        if  i>0:
            abline_values2.append(i)

    plt.plot(abline_values2,'-o',label='Intercepe:' + str(round(intercept,1)) + 'nm' )


def distrbuitionmodel(y):
    x=range(len(y))
    mod = LognormalModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)

    plt.plot(out.best_fit,'-o',label='distribuition')


d1=getcenter(x1,y1)
d2=getcenter(x2,y2)


def removezero(y):
    for i in range(len(y)):
        if y[i]<0:
            y[i]=0
    return y

def normalizar(y):
    maximo=max(y)
    for i in range(len(y)):
        y[i]=y[i]/maximo
    return y

def removerbackground(x,y,m=5):
    y=normalizar(y)
    minimo= mean( sort(y)[:10])
    for i in range(len(y)):
        y[i]=y[i]-minimo
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.append(x[:m],x[-m:]),np.append(y[:m],y[-m:]))
    abline_values = [slope * i + intercept for i in x]
    return y-abline_values


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

    x1=np.array([0]*30)

    if pico1<pico2:
        y1 =np.concatenate((x,y1))
    else:
        y2 =np.concatenate((x,y2))



##    y22 =np.concatenate((x1,y2))
##    y22 =np.concatenate((y22,x1))
##    y11 =np.concatenate((x1,y1))
##    y11 =np.concatenate((y11,x1))

    return y1,y2


def fftproblem(y):
    y=savgol_filter(y,51,9)
    return y

def stokes(yn,ym,x1):
    global lambida
    yn=np.fft.rfft(yn)
    ym=np.fft.rfft(ym)
##    yn=fftproblem(yn)
##    ym=fftproblem(ym)
    newvetor=[]
    newL=[]
    #lambida=0.154

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


y1=removerbackground(x1,y1)
y11=removerbackground(x11,y11)
y2=removerbackground(x2,y2)
y22=removerbackground(x22,y22)

y1,y11=centralizar(y1,y11)
y2,y22=centralizar(y2,y22)

y3,x3=stokes(y1,y11,x1)
y4,x4=stokes(y2,y22,x2)

manv=50
plt.figure(1)
plt.subplot(L,C,P);P=P+1
plt.plot(y1)
plt.plot(y11)

plt.subplot(L,C,P);P=P+1
plt.plot(y2)
plt.plot(y22)

plt.subplot(L,C,P);P=P+1
plt.xlabel('$L(nm)$')
plt.ylabel("$L_A(nm)$")
plt.plot(x3[:manv],y3[:manv],'-o')
p7= polyfit(x3[:manv],y3[:manv],9)
plt.plot(x3[:manv],polyval(p7,x3[:manv])[:manv],'-o',label='002')
plt.legend()
plt.subplot(L,C,P);P=P+1
plt.plot(x4[:manv],y4[:manv],'-o')
p8= polyfit(x4[:manv],y4[:manv],9)
plt.plot(x4[:manv],polyval(p8,x4[:manv])[:manv],'-o',label='004')
plt.xlabel('$L(nm)$')
plt.ylabel("$L_A(nm)$")
plt.legend()



vetorlen=range(2,100)
vetor1=[]
vetor3=[]
for i in vetorlen:
    vetor1.append(np.log(polyval(p7,i)))
    vetor3.append(np.log(polyval(p8,i)))


d1=lambida/(2*np.sin ( np.radians( d1/2 )))
d2=lambida/(2*np.sin ( np.radians( d2/2 )))

xx1=np.array([1/d1**2]*len(vetor1))
xx3=np.array([1/d2**2]*len(vetor3))

slope=[]
intercep=[]
mod = LinearModel()

vetor=[1,5,10,15,20,25,30,35,40,45,50,55,60]
plt.subplot(L,C,P);P=P+1

for i in range(80):
    xxplot=[xx1[i],xx3[i]]
    yyplot=[vetor1[i],vetor3[i]]
    #if  i in vetor:
    plt.plot(xxplot,yyplot,'-o')

    x=xxplot
    y=yyplot
    try:
        pars = mod.guess(y, x=x)
        out  = mod.fit(y, pars, x=x)


        print i,'microdeformacao: ', abs((out.values['slope']/(-2*pi**2)))

        slope.append(  ( out.values['slope']/(2*pi**2)))
        intercep.append( ( out.values['intercept'] ) )
    except:
        pass




plt.xlabel('$1/d^2(nm^2)$')
plt.ylabel("$LN(L_A(nm))$")



for i in range(len(intercep)):
    intercep[i]=np.e**intercep[i]
    if intercep[i]<0:
        intercep[i]=0
##    slope[i]=np.e**slope[i]
    if slope[i]<0:
        slope[i]=0



plt.subplot(L,C,P);P=P+1
plt.plot(intercep,'-o')
interceptarwarren(intercep)
plt.legend()
plt.grid()
plt.xlabel('$L(nm)$')
plt.ylabel("$L_v(nm)$")


derivada=np.gradient(np.gradient(intercep))
for i in range(len(derivada)):

    if derivada[i]<=0:
        derivada[i]=0

derivada=normalizar(derivada)
np.savetxt('test.out', derivada, delimiter=',')
plt.subplot(L,C,P);P=P+1
plt.plot(derivada,'-o')
distrbuitionmodel(derivada)
plt.legend()
plt.grid()
plt.title('distuibution')

plt.xlabel('$L(nm)$')
plt.ylabel("$Distribution$")


plt.subplot(L,C,P);P=P+1
plt.title('Stress')
plt.xlabel('$L(nm)$')
plt.ylabel("$RMSS$")
plt.plot(slope,'-o')
plt.legend()
plt.grid()
plt.show()

