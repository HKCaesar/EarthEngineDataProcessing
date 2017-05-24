# EarthEngine

(Data collection and preprocessing for machine learning on time series data generated from historical satellite images)

The scripts folder contains the main algorithms used for data preprocessing. Run earth_engine_script.js on Google Earth Engine to download image, mask and metadata to google drive. Define a geometry in Earth Engine and supply it to the first line of the script to select the region to be in the output. The "testPoint" in line 2-3 is only used to map centering and can be commented out.

read_land_use_no_gdal.py is the module for label map generation. The function read_land_use() reads a selected region (in WKT format, can be inferred from the selected region coordinates in Earth Engine) and desired target resolution, filters data in the shapefile database and generates a claa label map for the selected region.

read_image_data_scaleable.py is the script for raster data preprocessing and training set generation. The function old_data_preprocess_workflow() executes all the steps needed to generate a training data set.

Use the Instructions.ipynb notebook for further instructions and testing. If running on a clean installation, install Anaconda (Python 3) and the packages rasterio, fiona, shapely (preferably with *conda install*) and run *jupyter notebook* in the root directory to view and run the notebook.
