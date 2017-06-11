from model import Model

# Moments of  model (All):
PSI = [0.41, 0.39, 0.27]                         # Labor supply elasticity
CHI = [0.36, 0.37, 0.44]                         # Disutility of labor supply

for group in [0]:
    m = Model(psi=PSI[group], chi=CHI[group], group=group)
    m.evaluate_model()
    m.save_to_csv(group + 1)