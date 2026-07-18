# PolarVis

PolarVis provides a user interface for calibrating polarizerd filter array (PFA) cameras, processing the images, and creating visualizations of the results. It is developed as a companion to an Honours Thesis "Processing and Visualization Software for Polarized Filter Array Cameras" by Juliusz Ameljan-Kowalski. The report can be found on the [TU Delft repository](https://resolver.tudelft.nl/uuid:d61bab45-9071-444e-aa40-fddbd7df9705).


## Version

This is version `1.0.0`. PolarVis utilizes semantic versioning (see [semver.org](https://semver.org/)). The graphical user interface of PolarVis is considered its public API.


## Features

- PFA Camera calibration support
- Polarization image reconstruction
- Stokes parameter calculation
- Intensity, DoLP and AoP visualization
- Interactive visualization and data feedback
- Batch processing
- Export to NumPy and MATLAB formats

## Installation

Before installing PolarVis, make sure you have Python 3.11 or newer. Using Git, run: <br>
`git clone https://github.com/Julek-AK/PolarVis` <br>
`cd PolarVis`

You can use any other method of cloning a repository, or alternatively download the project as a ZIP file from GitHub and extract it.

Verify that your preferred Python instance satisfies all requirements in the `requirements.txt` file. Otherwise, create a dedicated virtual environment as such:

Windows: <br>
`python -m venv .venv` <br>
`.venv\Scripts\activate`

Linux / macOS:  <br>
`python3 -m venv .venv` <br>
`source .venv/bin/activate`

Install all project dependencies: <br>
`python -m pip install --upgrade pip` <br> 
`python -m pip install -r requirements.txt`

Run the application from project root: <br>
`python polarvis.py`

PolarVis uses PyTorch for image processing, and GPU acceleration is enabled by default (this can be changed in settings). If a compatible NVIDIA GPU is detected, CUDA will be used. Otherwise, PolarVis will fall back to CPU with no additional configuration.


## Simple Usage Guide

Navigate to the PolarVis folder and run `python polarvis.py` using your preferred Python instance, ensuring all requirements are satisfied. If you don't have an instance set up, follow the steps in Installation. Give the software some time to start up. 

Once active, go to `Calibration -> Compute Calibration`. In the popup window, rename the calibration to "Sample Calibration" and change the `Angles [deg]` field to be: "1.5,46.5,91.5,136.5". Next, select folders for each of the image series from the `data\samples\sample_calibration_data` folder, which should be shipped together with PolarVis. There will be one image in each series. Run the calibration.

Refresh the calibration panel on the left, and select your new Sample Calibration. Go to `Processing -> Single Processing`, and in the opened dialog select one of the three sample images in the `data\samples` folder. Click `Process`. The image ID should now be visible in the cache selection on the right. Click on it, and then select `Full Polarimetric Colormap`. Save the visualization to the `samples` folder as well, making sure to include image legend. Finally, go to `File -> Export Data`. Select your cached ID, the save location in the `samples` folder, the Stokes Parameter representation and `MAT` file format. Press `OK` to export scientific data.


## Supported Cameras
The PFA camera used during development is a BFS-U3-51S5P-C Blackfly Polarized Monochrome Camera. The actual photo-sensor array in the camera is a IMX250MZR/MYR sensor from Sony. Any camera with the same sensor should be compatible, and other PFA cameras with a 2x2 micro-polarizer grid should work as well.


## Settings
It is possible for you to edit some settings of the software. In general, it is good practice to restart PolarVis after you've edited settings (automatic restarting is not implemented as of `1.0.0`). If you believe you edited some settings in a way that causes serious misbehaviour, deleting the `user_settings.json` file from `polarvis\app\config` should force-restore the defaults. Some settings of note include GPU acceleration (if your system has CUDA available), selecting which colormap is used for which visualization, and configuring default directories to open when selecting files.


## Calibration
To calibrate the camera, use the option `Calibration -> Compute Calibration`. Note that as of `1.0.0`, only one calibration model is implemented, as described in the accompanying report. A dialog window will open, in which you can fill out all typed fields. Typing out comma seperated angle values in degrees will automatically create more data series input fields. For each one, select the folder that contains all images from the series. Keep in mind that running the calibration can take quite a long time.

The result of calibration is a standalone `.calib` file. It contains some metadata and raw data arrays used for processing images. You can preview your calibrations and select the one to be used for processing in the menu on the left side. The default calibration should always be available, and is computed on startup using default camera configuration from settings.


## Processing Pipeline
As of 1.0.0, the two working processing modes are single processing and batch processing. Video processing in unfortunately not available due to problems with video file streaming that could not have been resolved within the publication timeframe. Single processing lets you select a file to be processed, while batch processing lets you select a folder, and then attempts to process every image file in that folder. The results are processed using the selected calibration file, and then saved into the cache. Note that since the cache stores uncompressed scientific results, it can easily grow to multiple gigabytes in size.


## Visualizations
To generate a visualization, select a result from the cache in the visualization window on the right. Then, select the desired visualization mode. You can pan and zoom in the image viewing window. When you hover your mouse over a pixel, the polarized parameters corresponding to that pixel will be displayed in the information panel in the bottom-right. You can also save the visualization image, with or without a legend. Note that this outputs a standard .png image, which should **not** be parsed to extract polarized data. Raw data for scientific purposes should instead be obtained through exporting.


## Export
To export scientific-grade results, simply use the option `File -> Export Data`. In the pop-up window, select all cached results you wish to export and the desired target directory. All results will be exported into that folder. Select whether you'd like to save the result as Stokes parameters (S~0~, S~1~, S~2~) or as polarized description (I, DoLP, AoP). Finally, select the file format you'd like to use.


## Development & AI Statement
PolarVis was developed as a side project, as a part of the Honours Programme at TU Delft. Third-party Large Language Models were used during development, though only as chatbots, not as coding agents. The author made a great effort to write all code by hand where possible, and to otherwise understand every design choice and line of code generated by an LLM. Still, the author is a very inexperienced developer, so code errors and poor designs are likely quite common in PolarVis. The author recognizes this shortcoming and expresses the willingness to learn and improve. 
