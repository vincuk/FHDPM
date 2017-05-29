JOBS = 2                          # Numbers of jobs
CRITERION = 1e-3                  # Convergence criterion
MAXITERATIONS = 100               # Maximal number of iterations
PERIODS = 20                      # Number of periods for simulation
AGENTS = 1000                     # Number of agents

AUGMA = 22000                    # Borrowing limit
NBA = 20                          # Number of assets points in the grid
AMIN = 1000                      # Minimal value of assets
AMAX = AMIN + 1.5*PERIODS*AUGMA   # Maximal value of assets
DELTAA = (AMAX - AMIN) / NBA      # Increment of assets

AUGMH = 4320                      # Maximum hours worked per year
NBH = 20                          # Number of hours worked points in the grid
HMIN = 1000                       # Minimal value of hours worked
HMAX = HMIN + 1.5*PERIODS*AUGMH   # Maximal value of hours worked
DELTAH = (HMAX - HMIN) / NBH      # Increment of hours worked