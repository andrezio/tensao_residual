#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     23/07/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from CifFile import ReadCif
cf = ReadCif('UO2_STAR_246851.cif')
lambida=1.54
chave = cf.keys()


##for k,v in cf[chave[0]].items():
##    print k

value= cf[chave[0]]['_cell_length_a']

value=value.split('(')[0]

