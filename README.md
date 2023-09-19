# PSNet

<img src="https://github.com/jaschers/psnet/blob/master/logo/logo.png" width="250">

PSNet is the first application of pattern spectra on convolutional neural networks (CNNs) for the event reconstruction of imaging atmospheric Cherenkov telescopes (IACTs). We train a CNN on pattern spectra of gamma-ray events from the Cherenkov Telescope Array (CTA) for energy reconstruction and signal-background separation. PSNet is based on Tensorflow 2.3.1 and Keras 2.4.3 and uses the [ctapipe](https://github.com/cta-observatory/ctapipe) software for the data handling. This project is part of my PhD at the University of Groningen. For more information, see the following publications:

- [J. Aschersleben, R. F. Peletier, M. Vecchi, M. H. F. Wilkinson (2021)](https://arxiv.org/abs/2108.00834)

- [J. Aschersleben, R. F. Peletier, M. Vecchi, M. H. F. Wilkinson (2023)](https://arxiv.org/abs/2302.11876)

## Installation

### dm-finder repository
Clone this repository into your prefered folder:
```sh
git clone git@github.com:jaschers/psnet.git
```

### Anaconda
Follow the [instructions](https://docs.anaconda.com/anaconda/install/linux/) to install ``Anaconda3``. 

### Environment setup
Setup the ``psnet`` environment:

```sh
conda env create -f environment.yml
```

Activate the ``psnet`` environment:

```sh
conda activate psnet
```
## Usage
Every script has a help option ``-h`` or ``--help`` in order to get basic instructions on how to use the script. Some details will be discussed in the following.

### CTA data
Go into the ``psnet``directory and use 
```sh
mkdir -p data/gamma/event_files data/gamma_diffuse/event_files data/proton/event_files
``` 
to create the ``event_files`` directories and move your corresponding CTA data into the directories. In our work, we used simulated gamma-ray and proton events detected by the southern CTA array (Prod5 DL1 ([ctapipe v0.10.5](https://github.com/cta-observatory/ctapipe)), zenith angle of 20 deg, North pointing, see [here](https://zenodo.org/record/6218687#.ZG9U9tZBzao) for more details).

### Extract CTA images
Run 
```sh
python main/iact_images/create_iact_images.py -h
```
to get basic instructions on how to use the script. This script saves simulated CTA (SST) images in an HDF table. The images can be saved as mono images or as a sum of all images for each inidividual event (stereo sum). 
Examples: 
```sh
python main/iact_images/create_iact_images.py -pt gamma -tm stereo_sum_cta
``` 
creates CTA images from gamma-ray events from the data runs listed in ``main/run_lists/gamma_run_list_alpha.csv`` in the ``stereo_sum_cta`` telescope mode (all individual telescope images of each event are summed up). CTA images from one particular run can be created by adding the ``-r`` command, e.g 
```sh
python main/iact_images/create_iact_images.py -pt gamma -tm stereo_sum_cta -r 100
``` 
will create CTA images from data run 100. The images are saved into the ``data/gamma/images`` directory.

### Pattern spectra

#### Extraction
```sh
python main/pattern_spectra/python/create_pattern_spectra.py -h
```

This script creates pattern spectra from the CTA images of gamma/diffuse-gamma/proton events. The pattern spectra characteristics can be specified with ``-a`` (attributes), ``-dl`` (domain lower), ``-dh`` (domain higher), ``-m`` (mapper), ``-n`` (size) and ``-f`` (filter).

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

The pattern spectra are saved as matlab files into the ``data/<particle_type>/pattern_spectra`` directory. Again the pattern spectra are created from the runs listed in ``main/run_lists/<particle_type>_run_list_alpha.csv``. Pattern spectra from a particular run can be created by adding the ``-r`` command.

### Evaluation / investigation
The total data set of pattern spectra can be further investigated with the following script:

```
python main/pattern_spectra/python/pattern_spectra_evaluation.py -h
```

It evaluates the pattern spectra pixel distribution for different energies and particle types. The evaluation plots will be saved under ``data/<particle_type>/info/pattern_spectra_distribution/<pattern_spectra_specifications>/``:

### Convolutional neural network (CNN)
The code provides options to train and evaluate a CNN for energy reconstruction of gamma rays, and for the separation of gamma-ray and proton events. 

#### Training
```
python main/cnn/cnn.py -h
```
It is highly recommened to train the CNN on a computer cluster, if the full data set is used for training. 

Tests can be performed with a smaller data set listed in ``main/run_lists/<particle_type>_run_list_alpha_test.csv`` on your local machine with the ``-sd y`` option. The mode argument ``-m`` and the input argument ``-i`` are required in order to run the script. Other optional arguments will be discussed in the following.

##### Signal/background separation
```
python main/cnn/cnn.py -m separation -i cta
python main/cnn/cnn.py -m separation -i ps
```
The CNN can be trained for signal/background (photon/proton) separation with the CTA images ``-i cta`` or the pattern spectra ``-i ps`` as input. The pattern spectra characteristics can be specified as described in the **Pattern spectra - Extraction** section. By default, the full data set of all runs listed in ``main/run_lists/gamma_diffuse_run_list_alpha.csv`` and ``main/run_lists/proton_run_list_alpha.csv`` are considered. The energy range of the considered gamma-ray and proton events can be specified with the ``-erg <energy_lower> <energy_upper>`` and ``-erp <energy_lower> <energy_upper>`` arguments. Currently, we recommend to use ``-erg 0.5 100`` and ``-erp 1.5 100`` to consider gamma-ray events between 500 GeV and 100 TeV and proton events between 1.5 TeV and 100 TeV. We recommend to always specify the ``-na <name>`` argument in order to give a name to the particular experiment. The training of the CNN is stopped if there is no improvement on the validation dataset for over 20 epochs, and the model with the lowest validation loss is saved.

##### Energy reconstruction
```
python main/cnn/cnn.py -m energy -i cta
python main/cnn/cnn.py -m energy -i ps
```
The CNN can be trained for energy reconstruction with the CTA images ``-i cta`` or the pattern spectra ``-i ps`` as input. The pattern spectra characteristics can be specified as described in the **Pattern spectra - Extraction**. By default, the full data set of all runs listed in ``main/run_lists/gamma_run_list_alpha.csv`` are considered. The energy range of the considered events can be specified with the ``-erg <energy_lower> <energy_upper>`` argument. Currently, we recommend to use ``-erg 0.5 100`` to consider events between 500 GeV and 100 TeV. We recommend to always specify the ``-na <name>`` argument in order to give a name to the particular experiment. The training of the CNN is stopped if there is no improvement on the validation dataset for over 20 epochs, and the model with the lowest validation loss is saved.

#### Evaluation
```
python main/cnn/cnn_evaluation.py -h
```
The CNN evaluation script loads the output csv file that contains the performance of the CNN on the test data and evaluates the results. 

##### Signal/background separation
```
python main/cnn/cnn_evaluation.py -m separation -i <ps/cta> -na <name> -erg <energy_lower> <energy_upper> -erp <energy_lower> <energy_upper>
```
Specify ``-m separation`` in order to evaluate a CNN that was trained for signal/background separation. Use the same ``<name>`` for the ``-na`` option and the same ``<energy_lower> <energy_upper>`` for the ``-erg`` option that you specified for the CNN training in the previous section. The gammaness limit ``-gl <g_min_gamma> <g_max_gamma> <g_min_proton> <g_max_proton>`` option is optional and can help to investigate wrongly classified events. The plots will be extracted and saved under ``cnn/<iact_images/pattern_spectra>/separation/results/<pattern_spectra_specifications>/<name>/``.

##### Energy reconstruction
```
python main/cnn/cnn_evaluation.py -m energy -i <ps/cta> -na <name> -erg <energy_lower> <energy_upper>
```
Specify ``-m energy`` in order to evaluate a CNN that was trained for energy reconstruction. Use the same ``<name>`` for the ``-na`` option and the same ``<energy_lower> <energy_upper>`` for the ``-erg`` option that you specified for the CNN training in the previous section. The plots will be extracted and saved under ``cnn/<iact_images/pattern_spectra>/energy/results/<pattern_spectra_specifications>/<name>/``.


It is also possible to directly compare the results of several CNNs, e.g. via
```
python main/cnn/cnn_evaluation.py -m energy -i cta ps -na <name_cta> <name_ps> -erg <energy_lower> <energy_upper>
```
The corresponding plots are saved under ``cnn/comparison/``. 

## Information for students of the University of Groningen
### Git and personal access token
Create an GitHub account [here](https://github.com/). Check if ``git`` is installed on the machine you are working on via ``git --version``. Setup git with the following commands:
```
git config --global user.name "<firstname> <lastname>"
git config --global user.email "<email>"
git config --list
```
Create a personal access token by following the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

### CNN training on a computer cluster
In order to copy your local data and main on the Peregrine HPC cluster, copy the following commands into your ``~/.bashrc`` file:

```
alias sshperigrine='ssh -X <your_P/S-number>@peregrine.hpc.rug.nl'
alias pushperegrine='rsync -avzu <path_on_your_local_machine>/main <path_on_your_local_machine>/cnn <your_P/S-number>@peregrine.hpc.rug.nl:/data/<your_P/S-number>/dm-finder'
alias pullperegrine='rsync -avzu <your_P/S-number>@peregrine.hpc.rug.nl:/data/<your_P/S-number>/cnn <path_on_your_local_machine>/dm-finder'
```
Source your ``.bashrc`` file to apply the updates via ``source ~/.bashrc``. The ``sshperigrine`` command allows you to connect to the Peregrine cluster. The ``pushperegrine`` copies your data and main on the Peregrine cluster. The ``pullperegrine`` command copies the output of your CNN training on your local machine. In order to run a script on the Peregrine cluster, follow the steps below:
1. Login to the Peregrine cluster via ``sshperigrine``
2. Go into your working directory via ``cd /data/<your_P/S-number>``
3. Create a folder for your jobs via ``mkdir jobs``
4. Create a folder for your output via ``mkdir outputs``
5. Install Keras via ``pip install keras==2.4.3 --user``
6. Install Tables via ``pip install tables --user ``
7. Install TQDM via ``pip install tqdm --user ``
8. Load the required modules via ``module add matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4`` and ``module add TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4``
9. Save the modules for later use ``module save psnet``
10. Create your job, e.g. with vim via ``vim jobs/<name_of_your_job>.sh``
11. Copy the following lines into the file (this is an example of a job, you have to adjust it according to your needs):

```
#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=60G
#SBATCH --job-name=cta_s
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=j.j.m.aschersleben@rug.nl
#SBATCH --output=outputs/cta_images_cnn_separation.log
module restore psnet
python /data/<your_s_number>/main/cnn/cnn.py -m separation -i cta -na 0.5_100_TeV_exp1 -er 0.5 100 -e 50
```
10. Close and save the file. 
12. Load your modules via ``module restore psnet``
13. Send a job request via ``sbatch jobs/<name_of_your_job>.sh``
14. You can check the current status of your job via ``squeue -u $USER``

More information can be found on the [Peregrine HPC cluster wiki page](https://wiki.hpc.rug.nl/peregrine/start). After the job is completed, you can copy the output of your neural network to your local machine via ``pullperegrine`` (on your local machine).
