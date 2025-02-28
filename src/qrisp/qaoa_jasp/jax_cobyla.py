import jax
import jax.numpy as jnp
import time


def cobyla(func, x0, 
           #func_args, 
           #CHANGE
           cons=[], rhobeg=1.0, rhoend=1e-6, maxfun=100, 
           #callback=False
           ):

    """
    Optimizer for Jaspified qrisp Code. Mimics the scipy.cobyla optimizer
    
    Parameters
    ----------
    func: function
        the function to optimize
    x0: jnp.array
        inital values
    rhobeg: float
        inital rho
    rhobeg: float
        final rho
    maxfun: int
        maximal number of function iterations

    Returns
    -------
    sim[best], f[best]:
        The optimal result after running the optimization. The first value represents to final optimization paramaters, the second the final function value.

    """
    #func = func_in(func_args)

    n = len(x0)
    m = len(cons)
    #CHANGE
    callback_list=jnp.zeros(shape=(maxfun,))
    start_time = time.time()
    
    # Initialize the simplex
    sim = jnp.zeros((n + 1, n))
    sim = sim.at[0].set(x0)
    sim = sim.at[1:].set(x0 + jnp.eye(n) * rhobeg)
    
    # Initialize function values and constraint values
    f = jax.lax.map(func, sim)
    c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
    
    def body_fun(state):
        #CHANGE --> add callback_list
        sim, f, c, rho, nfeval = state
        
        # Find the best and worst points
        best = jnp.argmin(f)
        worst = jnp.argmax(f)
        
        # Calculate the centroid of the simplex excluding the worst point
        # Calculate the centroid of the simplex excluding the worst point
        mask = jnp.arange(n + 1) != worst
        centroid = jnp.sum(sim * mask[:, None], axis=0) / n
        
        # Reflect the worst point
        xr = 2 * centroid - sim[worst]
        fr = func(xr)
        cr = jnp.array([con(xr) for con in cons])
        nfeval += 1
        
        # Expansion
        xe = 2 * xr - centroid
        fe = func(xe)
        ce = jnp.array([con(xe) for con in cons])
        nfeval += 1
        
        # Contraction
        xc = 0.5 * (centroid + sim[worst])
        fc = func(xc)
        cc = jnp.array([con(xc) for con in cons])
        nfeval += 1
        
        # Update simplex based on conditions
        cond_reflect = (fr < f[best]) & jnp.all(cr >= 0)
        cond_expand = (fe < fr) & cond_reflect
        cond_contract = (fc < f[worst]) & jnp.all(cc >= 0)
        
        sim = jnp.where(cond_expand, sim.at[worst].set(xe), 
                jnp.where(cond_reflect, sim.at[worst].set(xr), 
                    jnp.where(cond_contract, sim.at[worst].set(xc), 
                        0.5 * (sim + sim[best]))))
        
        # f = jax.vmap(func)(sim)
        f = jax.lax.map(func, sim)
        c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
        #CHANGE --> uncomment
        #callb = callb.at[nfeval].set(f[best])
        #callb[nfeval] = f[best]

        rho *= 0.5
        #CHANGE --> uncomment
        return sim, f, c, rho, nfeval #, callb
    
    def cond_fun(state):
        #CHANGE --> add callback_list
        _, _, _, rho, nfeval = state
        return (rho > rhoend) & (nfeval < maxfun)
    
    # Main optimization loop
    #CHANGE --> add callback_list to inital state as final argument
    state = (sim, f, c, rhobeg, n + 1) # (sim, f, c, rhobeg, n + 1, callback_list)
    state = jax.lax.while_loop(cond_fun, body_fun, state)

    #CHANGE --> add callback_list
    sim, f, _, _, _ = state
    best = jnp.argmin(f)

    #CHANGE --> uncomment
    #final_time = start_time-time.time()

    #CHANGE --> add final_time and callback_list as returns
    #if callback:
        #return sim[best], f[best], final_time, callback_list
    #else
    return sim[best], f[best]


