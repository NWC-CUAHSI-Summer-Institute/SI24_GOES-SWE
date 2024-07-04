"""
# GOES data stacking

## Author(s)
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 2024-06-20
Description: This script stacks the GOES 16 data for daylight hours in California and Radiation variable in a Zarr file. It also can resample the data before saving it. 
## License
This software is licensed under the MIT. See the LICENSE file for more details.
"""

# %%
import os
import xarray as xr
from tqdm import tqdm
import time
import dask.array as da
import numpy as np

# Get the start time
print(f"Start Time: {int(time.time() // 3600)} hours")
start_time = time.time()

def data_stack(year=None, band=None, write_path=None, read_path=None, resample=False, resample_time=None):
    """
    Function to stack GOES 16 data for daylight hours in California and Radiation variable in a Zarr file.
    Optionally resamples the data before saving.

    Parameters:
    - year (str): The year of the data.
    - band (str): The band of the GOES data.
    - write_path (str): The path where the Zarr file will be saved.
    - read_path (str): The path where the GOES data is read from.
    - resample (bool): Whether to resample the data.
    - resample_time (str): The time frequency to resample the data.
    """
    print(f'Data Stacking for band {band} and year {year}')
    
    # Path to the parent folder containing the day folders
    parent_folder_path = read_path

    # Initialize an empty list to hold the combined "Rad" DataArrays for each day
    combined_rad_list = []

    # Loop through each day folder
    for day_folder in tqdm(os.listdir(parent_folder_path)):
        day_folder_path = os.path.join(parent_folder_path, day_folder)
        print(day_folder)
        
        if not os.path.isdir(day_folder_path):
            continue
        
        # Loop through each hour folder within the day folder
        for hour_folder in os.listdir(day_folder_path):
            hour_folder_path = os.path.join(day_folder_path, hour_folder)
            
            if not os.path.isdir(hour_folder_path):
                continue
            
            if (15 <= int(hour_folder) <= 23):  # Filter for daylight hours
                # List all NetCDF files in the current hour folder
                netcdf_files = [f for f in os.listdir(hour_folder_path) if f.endswith('.nc')]
                
                # Initialize an empty list to hold the "Rad" DataArrays for the current hour
                rad_list = []

                # Loop through the NetCDF files and open each one
                for file in netcdf_files:
                    file_path = os.path.join(hour_folder_path, file)
                    dataset = xr.open_dataset(file_path, chunks={'time': 10})  # Use Dask by specifying chunks
                    
                    # Extract the "Rad" variable
                    rad_variable = dataset['Rad']
                    
                    # Append the "Rad" DataArray to the list for the current hour
                    rad_list.append(rad_variable)
                
                # Concatenate the "Rad" DataArrays along a new time dimension for the current hour
                if rad_list:
                    combined_rad_hour = xr.concat(rad_list, dim='time')
                    combined_rad_list.append(combined_rad_hour)

    # Concatenate all hourly combined "Rad" DataArrays into a single DataArray
    if combined_rad_list:
        final_combined_rad = xr.concat(combined_rad_list, dim='time')

        if resample:
            # Resample the data if requested
            final_combined_rad = final_combined_rad.sortby('t')
            final_combined_rad = final_combined_rad.isel(time=np.delete(np.arange(final_combined_rad.sizes['time']), 0))

            # Rename the coordinate if necessary to avoid conflicts
            final_combined_rad = final_combined_rad.rename({'t': 'time'})

            # Sort the dataset by the time coordinate
            final_combined_rad = final_combined_rad.sortby('time')
            # Resample the dataset over the time dimension (e.g., monthly mean)
            final_combined_rad = final_combined_rad.resample(time=f'1{resample_time}').mean()

        # Save the final combined DataArray to a Zarr file
        output_file = f'{write_path}{band}_{year}.zarr'
        final_combined_rad.to_zarr(output_file, mode='w')
        print(f"\nFinal Combined 'Rad' DataArray saved to {output_file}")
    else:
        print("No 'Rad' data found in the specified folders.")
    
    # Print the total run time
    print(f"Run Time: {int((time.time() - start_time) // 3600)} hours, {int(((time.time() - start_time) % 3600) // 60)} minutes")


# %%

if __name__ == '__main__':
    # Read environment variables for configuration
    band = os.getenv('BAND')
    year = os.getenv('YEAR')
    write_path = os.getenv('WRITEPATH')
    resample = os.getenv('RESAMPLE') == 'True'
    resample_time = os.getenv('RESAMPLETIME')
    read_path = f'/home/shared/GOES_DA/band{band}/{year}'

    # Call the data_stack function with the provided parameters
    data_stack(year, band, write_path, read_path, resample, resample_time)
