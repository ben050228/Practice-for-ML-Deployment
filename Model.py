from pycaret.regression import *
from pycaret.datasets import get_data

import pandas as pd

data = pd.read_csv('Data/insurance.csv', index_col = False)

r2 = setup(data, target = 'charges', session_id = 123,
                                           normalize = True,
                                           polynomial_features = True, trigonometry_features = True,
                                           feature_interaction=True,
                                           bin_numeric_features= ['age', 'bmi'])

# Model Training and Validation
lr = create_model('lr')

# plot residuals of trained model
plot_model(lr, plot = 'residuals')

# save transformation pipeline and model
save_model(lr, model_name = 'deployment_ben')