"""
# GOES data Downloader

## Author(s)
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 2024-06-17
Description: This script downloads the GOES 16, 17, and 18 data using the goes2go package based on the user-specified properties.
## License
This software is licensed under the MIT. See the LICENSE file for more details.
"""

# Import the necessary modules
from goes2go import GOES
import time

def goes_get(start_date, end_date, save_path, satellite_name, product_name, domain_name, bands_name, cpus):
    # Record the start time to measure the execution time of the script
    start_time = time.time()

    # Initialize a GOES object with the specified parameters
    G = GOES(satellite=satellite_name, product=product_name, domain=domain_name, bands=bands_name)

    # Use the timerange method of the GOES object to retrieve data within the specified date range
    # and save it to the specified directory, utilizing the specified number of CPUs
    data_info = G.timerange(start=start_date, end=end_date, save_dir=save_path, max_cpus=cpus)

    # Print the total run time of the script in hours and minutes
    print(f"Run Time: {int((time.time() - start_time) // 3600)} hours, {int(((time.time() - start_time) % 3600) // 60)} minutes")

    # Return the data information retrieved by the GOES object
    return data_info

# Execute the main function if the script is run as the main module
if __name__ == "__main__":
    # Define the start and end dates for the data retrieval
    start_date = '2022-01-01 00:00'
    end_date = '2023-01-01 00:00'

    # Define the path where the retrieved data will be saved
    save_path = '../03.output/'

    # Specify the satellite, product, domain, and bands for the data retrieval
    satellite_name = 'goes16'
    product_name = "ABI-L1b-RadC"
    domain_name = 'C'
    bands_name = [1]

    # Set the number of CPUs to be used for the data retrieval process
    cpus = 18

    # Call the goes_get function with the specified parameters and store the returned data information
    data_information = goes_get(start_date, end_date, save_path, satellite_name, product_name, domain_name, bands_name, cpus)
