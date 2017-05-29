import numpy as np

BETA = 0.98                       # Discount rate
R = 0.04                          # Fixed interest rate
PSI = 0.5                         # Labor supply elasticity
IOTA = 0.5                        # Intertemporal elasticity of substitution
CHI = 0.43                        # Disutility of labor supply
TAU = 0.29                        # Marginal tax rate
WAGECONSTANT = 4.3                # Utility of being employed
ALPHA = -.45                      # Performance pay premium
XI = -0.11                        # Depreciation of human capital

# Learning by doing
GAMMA = [0.78, 0.07, 0.07] 

# Earnings premium
ZETA = [1, -.32, -.58, -.25, -.41, -.57, -.36, -.66, -.85] 

ZSHOCKS = 3                           # Number of discrete earnings shocks
z_shock_range = np.array([-2, 0, 2])  # Discrete earnings shocks range
pp_range = [0, 1]                     # List of possible PP states

# Probability of having PP given the job choice
PPDDISTRIBUTION = [0.039017, 0.065532, 0.065887, 0.096139, 0.088425, 0.298176, 
            0.106907, 0.134264, 0.105652]
            
# Transition matrix of discrete earnings shocks
PIMATRIX = np.array([0.1, 0.9, 0, 0.3, 0.4, 0.3, 0, 0.9, 0.1])
PIMATRIX = PIMATRIX.reshape(ZSHOCKS, ZSHOCKS)

# Probability of having discrete earnings shock
ZDISTRIBUTION = [0.2, 0.6, 0.2]

o_range = [False, True]                # List of possible o states