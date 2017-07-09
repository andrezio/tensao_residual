#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lgallego
#
# Created:     23/02/2017
# Copyright:   (c) lgallego 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from Tkinter import *
from ttk import *
import Pmw

import sys
import tkMessageBox
from tkFileDialog   import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from math import sin, cos
import numpy as np
from tkFileDialog   import askopenfilename
import copy
from lmfit.models import VoigtModel,PseudoVoigtModel, LinearModel
from math import sin,cos,pi,radians,tan,sqrt,log1p
from scipy import stats

K=0
E=1
v=1
psi=1

def Open_file():
    global x,y,x0,y0,namefile
    namefile = askopenfilename()

    try:
        x,y,z = np.loadtxt(namefile, unpack= True)
        x0=copy.copy(x[:])
        y0=copy.copy(y[:])

    except:
        x,y = np.loadtxt(namefile, unpack= True)
        x0=copy.copy(x[:])
        y0=copy.copy(y[:])


def Plotar():

    global x,y,K,E,v



    mod = VoigtModel()
    pars = mod.guess(y, x=x)
    pars['gamma'].set(value=0.7, vary=True, expr='')
    out  = mod.fit(y, pars, x=x)

    K=E/(1+v)
    K=K/2

    mult=np.tan(np.radians(out.best_values['center']/2))

    mult=1/mult


    print 'multi: ', mult

    K=K*mult*(-1)

    print K

    print out.best_values


    try:
        plt.title('Amostra')
        plt.xlabel('2Theta')
        plt.ylabel("Intensity")
        plt.plot(x, out.best_fit, 'r-',label='bestfit')
        plt.plot(x,y,linestyle='-', marker='o',label='material')
        plt.grid()
        plt.legend()
        plt.show()

    except:
        print 'vazio'


root = Tk()
root.title('Notebook')
Pmw.initialise()

nb = Pmw.NoteBook(root)
p1 = nb.add('SAMPLE')
p2 = nb.add('ANALYSIS')

nb.pack(padx=5, pady=5, fill=BOTH, expand=1)

#Sanple
texto = Label(p1,text='SHOW').place(x=10,y=5)

horizontal=0
vertical=40

btnPlotar = Button(p1, text="SAMPLE",command=Open_file).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p1, text="PLOT", command = Plotar).place(x=horizontal,y=vertical)
vertical+=30

ak=260
texto = Label(p1,text='CONSTANT').place(x=50,y=ak-30)
a=int
b=int
xc = Label(p1, text = "E")
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =0,y=ak )

bB = Entry(p1, textvariable = a)
bB.place(bordermode = OUTSIDE, height = 30, width = 40, x = 30, y =ak )

xd = Label(p1, text = "V")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =70,y=ak )

cC = Entry(p1, textvariable = b)
cC.place(bordermode = OUTSIDE, height = 30, width = 50, x = 100, y =ak )


#menu
menubar = Menu(root)
filemenu= Menu(menubar)
filemenu.add_command(label="Open File")
filemenu.add_command(label="Close")
filemenu.add_separator()

menubar.add_cascade(label="File",menu=filemenu)
helpmenu = Menu(menubar)
helpmenu.add_command(label="Help Index")
helpmenu.add_command(label="About")
menubar.add_cascade(label="Help",menu=helpmenu)
root.config(menu=menubar)

root.title("Cristal Mat - IPEN")
root.geometry("650x380+10+10")
root.mainloop()