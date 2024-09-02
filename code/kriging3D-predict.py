from pykrige.ok3d import OrdinaryKriging3D
import pickle
import pandas as pd
import numpy as np
from loadAndPreprocess import load_and_preprocess

'''
well_info: Well, X, Y, Total Resources
sensor_data: Depth, Porosity, Hydrate Saturation, Estimated Resources
'''
well_info, sensor_data_list = load_and_preprocess()

# Remove the data point with NaN value
well_info = well_info.dropna()
for sensor_data in sensor_data_list:
    sensor_data.dropna(inplace=True)
    

def load_model(name: str)-> OrdinaryKriging3D:
    with open(f'models/ok3d_{name}.pkl', 'rb') as f:
        ok3d_poro = pickle.load(f)
    return ok3d_poro

# Define the grid for interpolation
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#  Requires 120GB of RAM for this grid
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
step_sizes = [60.0, 60.0, 6.0]

X = well_info['X'].values
Y = well_info['Y'].values
Z = [sensor_data['Depth'] for sensor_data in sensor_data_list]
Z = np.concatenate(Z)

grid_x = np.arange(min(X)-200, max(X)+200, step_sizes[0])
grid_y = np.arange(min(Y)-200, max(Y)+200, step_sizes[1])
grid_z = np.arange(min(Z)-10, max(Z)+10, step_sizes[2])

grid = (grid_x, grid_y, grid_z)
with open('data/grid.pkl', 'wb') as f:
    pickle.dump(grid, f)

### Kriging Interpolation of Porosity
ok3d_poro = load_model('poro')
t_poro, ss3d_poro = ok3d_poro.execute('grid', *grid)
with open('data/t_poro.pkl', 'wb') as f:
    pickle.dump(t_poro, f)
with open('data/ss3d_poro.pkl', 'wb') as f:
    pickle.dump(ss3d_poro, f)

### Kriging Interpolation of Hydrate Saturation
ok3d_hydr = load_model('hydr')
t_hydr, ss3d_hydr = ok3d_hydr.execute('grid', *grid)
with open('data/t_hydr.pkl', 'wb') as f:
    pickle.dump(t_hydr, f)
with open('data/ss3d_hydr.pkl', 'wb') as f:
    pickle.dump(ss3d_hydr, f)
    