# %%
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import warnings
warnings.filterwarnings("ignore")

class GOESDataProcessor:
    def __init__(self, band, year, resample=True, resample_step=None, read_path=None, swe_station_list=None, write_path=None):
        """
        Initialize the GOESDataProcessor with the specified parameters.

        Parameters:
        band (int): The band number of the GOES data.
        year (int): The year of the GOES data.
        resample (bool): Whether to resample the data.
        resample_step (str): The resampling step (e.g., 'D' for daily).
        read_path (str): The path to the data files.
        swe_station_list (list): List of SWE station identifiers.
        """
        self.band = band
        self.year = year
        self.resample = resample
        self.resample_step = resample_step
        self.read_path = read_path
        self.write_path = write_path
        self.swe_station_list = swe_station_list
        self.data = None
        self.subset_data = None
    
    def calc_latlon(self, ds, dataset_netcdf):
        """
        Calculate latitude and longitude coordinates for the dataset.
        The math for this function was taken from:
        https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm

        Parameters:
        ds (xarray.Dataset): The dataset to add latitude and longitude coordinates.
        dataset_netcdf (xarray.Dataset): The reference NetCDF dataset.

        Returns:
        xarray.Dataset: The dataset with latitude and longitude coordinates added.
        """
        try:
            x = ds.x  # Get the x coordinates from the dataset
            y = ds.y  # Get the y coordinates from the dataset
            goes_imager_projection = dataset_netcdf.goes_imager_projection  # Get the projection information

            # Create a meshgrid for x and y
            x, y = np.meshgrid(x, y)

            # Extract projection attributes
            r_eq = goes_imager_projection.attrs["semi_major_axis"]
            r_pol = goes_imager_projection.attrs["semi_minor_axis"]
            l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi / 180)
            h_sat = goes_imager_projection.attrs["perspective_point_height"]
            H = r_eq + h_sat

            # Calculate latitude and longitude
            a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
            b = -2 * H * np.cos(x) * np.cos(y)
            c = H**2 - r_eq**2
            r_s = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            s_x = r_s * np.cos(x) * np.cos(y)
            s_y = -r_s * np.sin(x)
            s_z = r_s * np.cos(x) * np.sin(y)
            lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x)**2 + s_y**2))) * (180 / np.pi)
            lon = (l_0 - np.arctan(s_y / (H - s_x))) * (180 / np.pi)

            # Assign the calculated lat/lon to the dataset
            ds = ds.assign_coords({
                "lat": (["y", "x"], lat),
                "lon": (["y", "x"], lon)
            })
            ds.lat.attrs["units"] = "degrees_north"
            ds.lon.attrs["units"] = "degrees_east"

            return ds
        except Exception as e:
            print(f"Error in calc_latlon: {e}")
            return None

    def get_xy_from_latlon(self, ds, lats, lons):
        """
        Convert latitude and longitude bounds to x and y coordinates for slicing the data.

        Parameters:
        ds (xarray.Dataset): The dataset to extract x and y coordinates from.
        lats (tuple): Latitude bounds as (lat1, lat2).
        lons (tuple): Longitude bounds as (lon1, lon2).

        Returns:
        tuple: Two tuples containing the min and max x and y coordinates.
        """
        try:
            lat1, lat2 = lats  # Unpack latitude bounds
            lon1, lon2 = lons  # Unpack longitude bounds
            lat = ds.lat.data  # Get latitude data from the dataset
            lon = ds.lon.data  # Get longitude data from the dataset
            x = ds.x.data  # Get x coordinates from the dataset
            y = ds.y.data  # Get y coordinates from the dataset

            # Create a meshgrid for x and y
            x, y = np.meshgrid(x, y)

            # Filter x and y coordinates based on lat/lon bounds
            x = x[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]
            y = y[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]

            return ((min(x), max(x)), (min(y), max(y)))
        except Exception as e:
            print(f"Error in get_xy_from_latlon: {e}")
            return None, None

    def data_collection(self):
        """
        Collect and optionally resample data from a specified Zarr file.
        """
        try:
            print(f'Band {self.band} and Year {self.year}')
            zarr_file = f'{self.read_path}{self.band}_{self.year}.zarr'  # Construct the Zarr file path
            ds = xr.open_zarr(zarr_file)  # Open the Zarr file

            if self.band in [1, 2]:
                # Rename the coordinate if necessary to avoid conflicts
                ds = ds.rename({'t': 'time'})

            ds = ds.sortby('time')  # Sort the dataset by time

            if self.resample:
                # Resample the dataset over the time dimension (e.g., monthly mean)
                ds = ds.resample(time=f'1{self.resample_step}').mean()

            self.data = ds
        except Exception as e:
            print(f"Error in data_collection: {e}")
            self.data = None

    def catchment_subset(self):
        """
        Subset the dataset to the catchment area.

        Returns:
        xarray.Dataset: The subsetted dataset.
        """
        try:
            catchment = gpd.read_file('/home/snaserneisary/01.projects/05.summer_institute/SI24_GOES-SWE/02.input/01.shape_file/mycatchment.shp')  # Read the catchment shapefile
            catchment = catchment.to_crs('EPSG:4326').dissolve()  # Convert the CRS and dissolve geometries
            netcdf_files = [f for f in os.listdir(f'/home/shared/GOES_DA/band{self.band}/2023/001/15/') if f.endswith('.nc')]  # List NetCDF files
            dataset_netcdf = xr.open_dataset(f'/home/shared/GOES_DA/band{self.band}/2023/001/15/{netcdf_files[0]}')  # Open a NetCDF file
            ds_new = self.calc_latlon(self.data, dataset_netcdf)  # Calculate lat/lon for the data

            geoms = np.array(catchment.geometry.bounds)[0]  # Get the catchment bounds
            lats = (geoms[1], geoms[3])  # Extract latitude bounds
            lons = (geoms[0], geoms[2])  # Extract longitude bounds
            (x1, x2), (y1, y2) = self.get_xy_from_latlon(ds_new, lats, lons)  # Get x/y coordinates from lat/lon bounds

            ds_new = ds_new.sel(x=slice(x1, x2), y=slice(y2, y1))  # Subset the dataset based on x/y coordinates

            self.subset_data = ds_new
        except Exception as e:
            print(f"Error in catchment_subset: {e}")
            self.subset_data = None

        return self.subset_data 

    def corr_value(self, subset=True):
        """
        Calculate the correlation value between the dataset and SWE data for each station.

        Parameters:
        subset (bool): Whether the data is a subset or the full CONUS data.

        Returns:
        xarray.DataArray: The correlation values.
        """
        try:
            for station in self.swe_station_list:
                swe_series = pd.read_csv(f'/home/snaserneisary/01.projects/05.summer_institute/SI24_GOES-SWE/02.input/02.snotel_data/{station}-SWE.csv')  # Read the SWE data for the station
                swe_series = swe_series.iloc[:, 1:]  # Remove the first column
                swe_series.rename(columns={'date': 'time', 'Snow water equivalent': 'SWE'}, inplace=True)  # Rename columns

                swe_series['time'] = pd.to_datetime(swe_series['time']).dt.normalize()  # Convert and normalize time
                swe_series['time'] = swe_series['time'].dt.tz_localize(None)  # Remove timezone information

                goes = pd.DataFrame({'time': self.subset_data['time'].values})  # Create a DataFrame for GOES data time
                goes['time'] = goes['time'].dt.tz_localize(None)  # Remove timezone information
                merged_df = pd.merge(swe_series, goes, on='time', how='inner')  # Merge SWE and GOES data on time

                time_steps = merged_df['time'].values  # Get the time steps
                resampled_ds_selected = self.subset_data.sel(time=time_steps)  # Select the data based on time steps

                time_series_da = xr.DataArray(swe_series['SWE'].values, coords={"time": swe_series['time'].values}, dims=["time"])  # Create a DataArray for SWE values
                broadcasted_time_series = time_series_da.broadcast_like(resampled_ds_selected)  # Broadcast SWE values to match the selected data

                resampled_ds_selected["swe"] = broadcasted_time_series  # Add SWE values to the dataset
                corr_val = xr.corr(resampled_ds_selected['Rad'], resampled_ds_selected['swe'], dim='time')  # Calculate the correlation
                location = 'subset' if subset == True else 'CONUS'  # Determine the location type
                print(location)

                # Save the correlation results to NetCDF and Parquet files
                corr_val.to_netcdf(f'{self.write_path}corr_{station}_{self.band}_{location}_{self.year}.nc')
                df_corr_cpsc = pd.DataFrame(corr_val.values)
                df_corr_cpsc.to_parquet(f'{self.write_path}corr_{station}_{self.band}_{location}_{self.year}.parquet')
        except Exception as e:
            print(f"Error in corr_value: {e}")

        return corr_val

    def process(self, subset=True):
        """
        Run the full processing pipeline.

        Parameters:
        subset (bool): Whether to subset the data to the catchment area.

        Returns:
        xarray.DataArray: The correlation values if the processing is successful.
        """
        self.data_collection()  # Collect the data
        if self.data is None:
            return
        if subset == True:
            self.catchment_subset()  # Subset the data to the catchment area
            if self.subset_data is None:
                return
        else:   
            self.subset_data = self.data  # Use the full dataset if not subsetting
        test_1 = self.corr_value(subset)  # Calculate the correlation values
        return test_1


if __name__ == '__main__':
    # Parameters
    swe_station_list = ['PSC', 'CSV', 'UTY']  # List of SWE station identifiers
    subset = os.getenv('SUBSET')  # Whether to subset the data to the catchment area
    read_path = os.getenv('READPATH')  # Path to the data files
    band = os.getenv('BAND') # GOES data band number
    year = os.getenv('YEAR') # Year of the GOES data
    write_path = os.getenv('WRITEPATH') # Writing path for results
    resample = os.getenv('RESAMPLE') # Whether to resample the data
    resample_step = os.getenv('RESAMPLETIME') # Resampling step (daily)


    # Initialize the processor with the parameters
    processor = GOESDataProcessor(band, year, resample, resample_step, read_path, swe_station_list, write_path)
    print(f'Band {band} and Year {year} subset {subset} resample {resample}')

    # Run the processing pipeline
    corr_val = processor.process(subset)
