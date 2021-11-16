# dm-finder
## Preamble
The goal of this project is to investigate the potential of pattern spectra for analyses based on Cherenkov telescope data. In the current state, the goal of the project is NOT to outperfom the standard algorithms used by the CTA, H.E.S.S., ... collaborations.

## Installation
Create an GitHub account [here](https://github.com/). Check if ``git`` is installed on the machine you are working on via ``git --version``. Setup git with the following commands:
```
git config --global user.name "<firstname> <lastname>"
git config --global user.email "<email>"
git config --list
```
Follow the [instructions](https://docs.anaconda.com/anaconda/install/linux/) to install ``Anaconda3``. Next, clone this repository into your prefered folder:

```sh
git clone https://github.com/jaschers/dm-finder.git
```

### Environment setup
Setup the ``ctapipe`` environment:

```sh
conda install mamba -n base -c conda-forge
mamba env create --file environment.yaml
```

Start the ``ctapipe`` environment:

```sh
conda activate ctapipe
```
## Usage
Every script has a help option ``-h`` or ``--help`` in order to get basic instructions on how to use the script. Some details will be discussed in the following.

### CTA data download
Use 
```sh
mkdir -p dm-finder/data/gamma/event_files
``` 
to create the ``event_files`` directory and download the CTA data with ``DiRAC`` into the ``event_files`` directory (see the [Checklist for CTA newcomers](https://github.com/jaschers/cta-newcomers) for details). 

### Create CTA images
Run 
```sh
python dm-finder/scripts/iact_images/create_iact_images.py -h
```
to get basic instructions on how to use the script. This script creates Cherenkov images of gamma/diffuse-gamma/proton events simulated for CTA. The images are saved in tif and pgm format and stored in a HDF table. One can choose between int8 and float64 images. 
Examples: 
```sh
python dm-finder/scripts/iact_images/create_iact_images.py -pt gamma -dt float64
``` 
creates float CTA images from gamma-ray events from the data runs listed in ``dm-finder/scripts/run_lists/gamma_run_list.csv``. CTA images from one particular run can be created by adding the ``-r`` command, e.g 
```sh
python dm-finder/scripts/iact_images/create_iact_images.py -pt gamma -dt float64 -r 100
``` 
will create CTA images from data run 100. The images are saved into the ``dm-finder/data/gamma/images`` directory.

### Create pattern spectra
This script creates pattern spectra from the CTA images of gamma/diffuse-gamma/proton events. One can create the pattern spectra from int8 or float64 CTA images. The pattern spectra characteristics can be specified with ``-a`` (attributes), ``-dl`` (domain lower), ``-dh`` (domain higher), ``-m`` (mapper), ``-n`` (size) and ``-f`` (filter).

The following attributes, filter and mapper are available
```sh
attr =  0 - Area (default) 
        1 - Area of the minimum enclosing rectangle 
        2 - Length of the diagonal of the minimum encl. rect. 
        3 - Area (Peri) 
        4 - Perimeter (Peri) 
        5 - Complexity (Peri) 
        6 - Simplicity (Peri) 
        7 - Compactness (Peri) 
        8 - Moment Of Inertia 
        9 - (Moment Of Inertia) / (Area*Area) 
        10 - Compactnes                          (Jagged) 
        11 - (Moment Of Inertia) / (Area*Area)   (Jagged) 
        12 - Jaggedness                          (Jagged)
        13 - Entropy 
        14 - Lambda-Max (not idempotent -> not a filter) 
        15 - Max. Pos. X 
        16 - Max. Pos. Y 
        17 - Grey level 
        18 - Sum grey levels 
filter = 0 - "Min" decision 
        1 - "Direct" decision (default) 
        2 - "Max" decision 
        3 - Wilkinson decision 
mapper = 0 - Area mapper 
        1 - Linear mapper 
        2 - Sqrt mapper 
        3 - Log2 mapper 
        4 - Log10 mapper
```

The pattern spectra are saved as matlab files into the ``dm-finder/data/gamma/pattern_spectra`` directory. Again the pattern spectra are created from the runs listed in ``dm-finder/scripts/run_lists/gamma_run_list.csv``. Pattern spectra from a particular run can be created by adding the ``-r`` command.

In order to use the GUI of the pattern spectra code to have a look at an individual pattern spectrum, one has to go into the ``xmaxtree`` directoy via ``cd dm-finder/scripts/pattern_spectra/xmaxtree`` and run ``./xmaxtree <filename>.pgm a 9, 0 dl 0, 0 dh 10, 10 m 2, 0 n 20, 20 f 3``. The input parameter can be adjusted according to your needs.

### Convolutional neural network (CNN)
Currently, the code provides options to train a CNN for energy reconstruction of gamma rays, and for the separation of gamma-ray and proton events. 

#### Training

