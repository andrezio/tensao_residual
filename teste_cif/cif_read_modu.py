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
<<<<<<< HEAD
cf = ReadCif('UO2_STAR_246851.cif')
lambida=1.54
=======
cf = ReadCif('u308.cif')

>>>>>>> f26a11b75e998002824160efa1c337672e8cb21b
chave = cf.keys()


##for k,v in cf[chave[0]].items():
##    print k

<<<<<<< HEAD
value= cf[chave[0]]['_cell_length_a']

value=value.split('(')[0]

=======
print cf[chave[0]]['_symmetry_equiv_pos_as_xyz']
>>>>>>> f26a11b75e998002824160efa1c337672e8cb21b
