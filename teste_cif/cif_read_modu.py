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
cf = ReadCif('u308.cif')

chave = cf.keys()


##for k,v in cf[chave[0]].items():
##    print k

print cf[chave[0]]['_symmetry_equiv_pos_as_xyz']