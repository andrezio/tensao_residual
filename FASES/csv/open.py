import numpy as np
import matplotlib.pyplot as plt
import pdb
from lmfit.models import GaussianModel,LinearModel,VoigtModel
listacsv=['50.csv','56.csv','61.csv','74.csv','85.csv']

Thetavetor=[(26,31),(58,60.0)]
##Thetavetor=[(26,31)]

#x,xy50,Ycalc,Diff,Gd2O3,UO2 = np.loadtxt('50.csv', delimiter=';')
lambidacuka2=0.1537400

globaldicionario={}


def calcsingleline(gamma,sigma,center,tipo,arquivo):
    centersize=np.cos(np.radians(center/2))
    centerstress=np.tan(np.radians(center/2))
    #fwhm
    sigma=sigma*2.358
    gamma=gamma*2

    #Integral Breath
    gamma=gamma*(np.pi/2)
    sigma=sigma*1.06440701943

    #size
    Lv=lambidacuka2/(gamma*centersize)
    #stress
    e=gamma/(4*centerstress)
    e=e**2


    try:
        globaldicionario[tipo].append((arquivo,Lv,e))
    except:
        globaldicionario[tipo]=[(arquivo,Lv,e)]



def singleline(x,y,tipo,arquivo):
##    pdb.set_trace()
    mod = VoigtModel()
    pars = mod.guess(y, x=x)
    pars['gamma'].set(value=0.7, vary=True, expr='')
##    pars['sigma'].set(value=0.7, vary=True, expr='')
    out  = mod.fit(y, pars, x=x)

    gamma=out.best_values['gamma']
    sigma=out.best_values['sigma']
    center=out.best_values['center']
    calcsingleline(gamma,sigma,center,tipo,arquivo)

    #print tipo,'gamma tamanha:', out.best_values['gamma'],'sigma deformation:', out.best_values['sigma']


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

    		if yy<0:yy=(yy+y[i])/8

    		novoy.append(yy)
    	except:
    		novoy.append(y[i])

    return novoy

def background(y):
    minimo=min(y)
    for i in range(len(y)):
        y[i]-=minimo
    return y

def getminmax(x):
    mini1=float(theta[0])
    maxi1=float(theta[1])
    getminimo=0
    getmaximo=0
    for i in range(0, len(x)):
        if(x[i]<=mini1):
            try:
                getminimo=x[i+1]
            except:
                getminimo=x[i]
        if(x[i]<=maxi1):
            if(x[i]==x[len(x)-1]):
                getmaximo=x[i]
            else:
                getmaximo=x[i+1]

    mini = np.searchsorted(x,getminimo)
    maxi = np.searchsorted(x,getmaximo)

    return mini, maxi

def normalizar(y):
    maximo=max(y)
    for i in range(len(y)):
        y[i]=y[i]/maximo
    return y


L=2
C=5
N=1

plt.figure(1)
for theta in Thetavetor:
    for arquivo in listacsv:
        x = np.loadtxt(arquivo, delimiter=';')

        tamanho = len(x)

        novoy=[]
        novoy2=[]
        novoy3=[]
        novoy4=[]
        novox=[]
        for i in range(tamanho):
            novox.append(float( x[i][0]))
            novoy.append(float( x[i][1]))
            novoy2.append(float( x[i][4]))
            novoy3.append(float( x[i][5]))
            novoy4.append(float( x[i][2]))


        mini,maxi=getminmax(novox)

        novox=novox[mini:maxi]

        novoy=normalizar(novoy[mini:maxi])
        novoy2=normalizar(novoy2[mini:maxi])
        novoy3=normalizar(novoy3[mini:maxi])
        novoy4=normalizar(novoy4[mini:maxi])

        novoy=background(novoy)
        novoy2=background(novoy2)
        novoy3=background(novoy3)
        novoy4=background(novoy4)

        novoy=removekalpha(novoy,novox)
        novoy2=removekalpha(novoy2,novox)
        novoy3=removekalpha(novoy3,novox)
        novoy4=removekalpha(novoy4,novox)

        novoy=normalizar(novoy)
        novoy2=normalizar(novoy2)
        novoy3=normalizar(novoy3)
        novoy4=normalizar(novoy4)

        [float(i) for i in novox]
        [float(i) for i in novoy]
        [float(i) for i in novoy2]
        [float(i) for i in novoy3]
        [float(i) for i in novoy4]

        novox=np.asarray(novox)
        novoy=np.asarray(novoy)
        novoy2=np.asarray(novoy2)
        novoy3=np.asarray(novoy3)
        novoy4=np.asarray(novoy4)

        print arquivo, theta
        singleline(novox,novoy,'amostra',arquivo)
        singleline(novox,novoy2,'Gd2O3',arquivo)
        singleline(novox,novoy3,'UO2',arquivo)
        singleline(novox,novoy4,'calc',arquivo)

        plt.subplot(L,C,N)
        plt.plot(novox,novoy,label='amostra')
        plt.plot(novox,novoy2,label='Gd2O3')
        plt.plot(novox,novoy3,label='UO2')
        plt.plot(novox,novoy4,label='calc')
        plt.title(arquivo)
        plt.grid()
        plt.legend()
        N=N+1



##plt.show()




