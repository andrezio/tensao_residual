#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     16/04/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

namefile = "E:\\FASES\\FASES PARA EXCEL\\50.txt"

x,y,z,w,a,b = np.loadtxt(namefile, unpack= True,delimiter=';')