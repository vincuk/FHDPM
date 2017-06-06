import numpy as np

# Parameters of model:

BETA = 0.98                           # Discount rate
R = 0.04                              # Fixed interest rate
IOTA = 0.5                            # Intertemporal elasticity of substitution
TAU = 0.29                            # Marginal tax rate

ZSHOCKS = 3                           # Number of discrete earnings shocks
z_shock_range = np.array([-1, 0, 1])  # Discrete earnings shocks range
pp_range = [0, 1]                     # List of possible PP states

# Probability of having PP given the job choice
PPDDISTRIBUTION = [0.039017, 0.065532, 0.065887, 0.096139, 0.088425, 0.298176, 
            0.106907, 0.134264, 0.105652]
            
# Transition matrix of discrete earnings shocks
PIMATRIX = np.array([0.1, 0.9, 0, 0.3, 0.4, 0.3, 0, 0.9, 0.1])
PIMATRIX = PIMATRIX.reshape(ZSHOCKS, ZSHOCKS)

# Probability of having discrete earnings shock
ZDISTRIBUTION = [0.2, 0.6, 0.2]

o_range = [False, True]               # List of possible o states