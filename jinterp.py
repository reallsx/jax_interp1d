import jax.numpy as jnp

class interp1d:
    '''
    Interpolate a 1-D function.

    only for linear interpolation.
    
    '''
    def __init__(self, xp, yp):
        if xp.shape[-1] != yp.shape[-1]:
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")
        self.xp = xp
        self.yp = yp

    def __call__(self, x):
        return jnp.interp(x, self.xp, self.yp)
    

