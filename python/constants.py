import numpy as np

# Parameters of model:

BETA = 0.98                               # Discount rate
R = 0.04                                  # Fixed interest rate
IOTA = 1.5                                # Intertemporal elasticity of substitution
TAU = 0.29                                # Marginal tax rate

ZSHOCKS = 3                               # Number of discrete earnings shocks
z_shock_range = np.array([-1, 0, 1])      # Discrete earnings shocks range
pp_range = [0, 1]                         # List of possible PP states
o_range = [False, True]                   # List of possible o states