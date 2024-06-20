![team_logo](./image/team_logo.webp)

# Introduction

This repository represents the **"Investigating SWE Predictive Capability Using GOES Bands and Convolutional Neural Network"** project. This project aims to leverage Geostationary Operational Environmental Satellite (GOES) data to make spatial predictions of snow water equivalent (SWE) in the southern Sierra Nevada mountains. By employing Convolutional Neural Networks (CNNs), we analyze visual and infrared satellite imagery to generate these predictions. This repository includes all necessary code for downloading and processing data, model development, and evaluation.

# Installation

To use this repository, please clone it and install the required packages using the *requirements.yml* file with the code below:

```
git clone https://github.com/NWC-CUAHSI-Summer-Institute/SI24_GOES-SWE.git
```
```
cd SI24_GOES-SWE
```
```
mamba env create -f requirments.yaml
```
Then, you should install the ipykernel and connect it to jupyter notebook. 

```
pip install --user ipykernel
```
```
python -m ipykernel install --user --name=goes_kernel
```

# Background

Southern Sierra Nevada snowpack is a major source of spring streamflow in California. Precise measurements of SWE are critical for making accurate streamflow predictions year to year, which assist water management agencies in their decision making. Ground observations, such as those provided by SNOTEL stations, are limited to a small number of point source locations. Thus, this creates a need for more representative spatial predictions of SWE in areas without SNOTEL sites.

Previous studies have developed models using observational and satellite data to predict SWE in various regions of the US, however, data from the GOES satellite has been underutilized in the past and remains a potential source of valuable information. This project aims to assess the feasibility of assimilating GOES imagery across the contiguous United States (CONUS) as inputs to a CNN model for predicting SWE in high elevation mountainous regions.
