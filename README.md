![team_logo](./image/team_logo.webp)

# Introduction

This repository represents the **"Investigating SWE Predictive Capability Using GOES Bands and Convolutional Neural Network"** project. This project aims to leverage Geostationary Operational Environmental Satellites (GOES) data to make spatial predictions of Snow Water Equivalent (SWE) in the Southern Sierra Nevada mountains. By employing Convolutional Neural Networks (CNNs), we analyze big satellite datasets to generate these predictions. This repository includes all necessary code for downloading and processing data, model development, and evaluation.

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

The Southern Sierra Nevada SWE is a major source of streamflow for water resources management in California. Accurate SWE data is critical for making precise streamflow predictions, which assist water planners in water allocation decisions. Ground observation data, such as the SNOTEL datasets, are limited to specific locations, creating a need for accurate spatial predictions in areas without SNOTEL sites.

Various studies have developed models using different inputs, such as observational and satellite data, to predict SWE in various regions of the US. The GOES satellite is a potential source of valuable information, but it has been underutilized in the past. This project aims to assess the feasibility of using GOES data across the contiguous United States (CONUS) as inputs to a CNN model for predicting SWE in different locations.