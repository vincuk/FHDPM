import model

# Moments of model (All):
PSI = 0.41                         # Labor supply elasticity
CHI = 0.36                         # Disutility of labor supply

m = model.Model(PSI, CHI, 0)
m.evaluate_model()

m.show(0,-1,1)