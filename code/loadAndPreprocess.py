import pandas as pd
import os

def load_and_preprocess() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    well_info = pd.read_csv('well-loc.tsv', sep='\t')

    # Path to the sensor data directory
    sensor_data_path = 'sensor-data'

    # List all TSV files in the directory
    sensor_data_files = [f for f in os.listdir(sensor_data_path) if f.endswith('.tsv')]

    # Sort the sensor_data_files list
    sensor_data_files.sort(key= lambda x: int(x.split('.')[0]))

    # Load and concatenate all sensor data files into one DataFrame
    sensor_data_list = [pd.read_csv(os.path.join(sensor_data_path, file), sep='\t',
                                    na_values="-9999") for file in sensor_data_files]

    # Reset the index of the well_loc DataFrame to Well, X, Y
    well_info.rename(columns={'井': 'Well'}, inplace=True)

    # Reset the index of the sensor data DataFrame to Depth, Porosity, Hydrate Saturation
    for idx, _ in enumerate(sensor_data_list):
        sensor_data_list[idx].columns = ['Depth', 'Porosity', 'Hydrate Saturation']


    # # Absolute the values of Porosity and Hydrate Saturation
    # for idx, sensor_data in enumerate(sensor_data_list):
    #     sensor_data['Porosity'] = sensor_data['Porosity'].abs()
    #     sensor_data['Hydrate Saturation'] = sensor_data['Hydrate Saturation'].abs()

    # Drop the rows containing values larger than 1.0 except for the Depth column
    for idx, sensor_data in enumerate(sensor_data_list):
        sensor_data = sensor_data[sensor_data['Porosity'] <= 1.0]
        sensor_data = sensor_data[sensor_data['Hydrate Saturation'] <= 1.0]
        sensor_data_list[idx] = sensor_data
        
    # Drop the rows containing values smaller than 0.0 except for the Depth column
    for idx, sensor_data in enumerate(sensor_data_list):
        sensor_data = sensor_data[sensor_data['Porosity'] >= 0]
        sensor_data = sensor_data[sensor_data['Hydrate Saturation'] >= 0]
        sensor_data_list[idx] = sensor_data
        

    def estimate_resource(sensor_data: pd.Series) -> float:
        """Estimate the resource at a given location based on sensor data"""
        # Get the Porosity and the Hydrate saturation
        porosity = sensor_data['Porosity']
        hydrate_saturation = sensor_data['Hydrate Saturation']

        valid_volume = 0.1 # Assume the valid volume is 0.1 cubic meter
        factor = 155 # Assume the factor is 155

        # Calculate the resource estimate
        return valid_volume * porosity * hydrate_saturation * factor


    # Calculate the resource estimate for each sensor data in each depth
    for idx, sensor_data in enumerate(sensor_data_list):
        sensor_data_list[idx]['Estimated Resources'] = sensor_data.dropna().apply(estimate_resource, axis=1)

    sensor_data_list[0]

    resource_sums = [df['Estimated Resources'].sum() for df in sensor_data_list]
    well_info['Total Resources'] = resource_sums
    
    return well_info, sensor_data_list