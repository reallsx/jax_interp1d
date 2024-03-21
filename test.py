import jax.numpy as jnp
import matplotlib.pyplot as plt

from jinterp import interp1d



x = jnp.arange(0,10)
y = jnp.exp(-x/3.0)
f = interp1d(x,y)

xnew = jnp.arange(0,9,0.1)
ynew = f(xnew)
plt.plot(x,y, 'o',xnew, ynew, '-')
plt.show()
