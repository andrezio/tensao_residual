import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel
x,y = np.loadtxt('outsampleuo2_222.xy', unpack= True)

minimo=min(y)
y[:]=[ (i - minimo)  for i in y]


mod = VoigtModel()
pars = mod.guess(y, x=x)
pars['gamma'].set(value=0.3, vary=True, expr='')
out  = mod.fit(y, pars, x=x)


print(out.fit_report(min_correl=0.25))



plt.plot(x, y)
plt.plot(x, out.init_fit, 'k--')
plt.plot(x, out.best_fit, 'r-')
plt.show()