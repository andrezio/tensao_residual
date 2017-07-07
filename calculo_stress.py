import numpy as np
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

print theta #Mpq/deg
print theta/9.8#kg/mm2/deg

#Multiplicar pelo slope
