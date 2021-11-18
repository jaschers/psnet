# dm-finder
## Preamble
This project investigates the potential of pattern spectra for analyses based on Cherenkov telescope data. By applying pattern spectra on convolutional neural networks (CNNs), the goal of this project is to outperform the performance of the current standard analysis and/or to reduce the computational power needed to train the CNNs.

## Installation
### Git and personal access token
Create an GitHub account [here](https://github.com/). Check if ``git`` is installed on the machine you are working on via ``git --version``. Setup git with the following commands:
```
git config --global user.name "<firstname> <lastname>"
git config --global user.email "<email>"
git config --list
```
<!---
```
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
vim ~/.ssh/config
```
Add the following lines into the ``~/.ssh/config`` file:
```
Host *
  IgnoreUnknown UseKeychain
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```
Close the file with the ``esc``-key and type ``:wq`` followed by the ``enter``-key. Add your SSH private key to the ssh-agent and store your passphrase in the keychain:

```
ssh-add -k ~/.ssh/id_ed25519
```
Ope the the ssh key with ``vim ~/.ssh/id_ed25519.pub`` and copy the content of the file. Go on [GitHub](https://github.com/) -> click your profile photo -> Settings -> SSH and GPG keys -> New SSH key or Add SSH key. In the "Title" field, add a descriptive label for the new key. Paste your key into the "Key" field. Click 'Add SSH key'. If prompted, confirm your GitHub password. Test your ssh connection with ``ssh -T git@github.com``. If everything was setup correctly, you should get the following message

```
You've successfully authenticated, but GitHub does not provide shell access.
```
--->
Create a personal access token by following the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

### dm-finder repository
Clone this repository into your prefered folder:
```sh
git clone https://github.com/jaschers/dm-finder.git
```

### Anaconda
Follow the [instructions](https://docs.anaconda.com/anaconda/install/linux/) to install ``Anaconda3``. 

### Environment setup
Setup the ``ctapipe`` environment:

```sh
conda install mamba -n base -c conda-forge
mamba env create --file environment.yml
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
mkdir -p dm-finder/data/gamma_diffuse/event_files
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
```sh
python dm-finder/scripts/pattern_spectra/python/create_pattern_spectra.py -h
```

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
Currently, the code provides options to train and evaluate a CNN for energy reconstruction of gamma rays, and for the separation of gamma-ray and proton events. 

#### Training
```
python dm-finder/scripts/cnn/cnn.py -h
```
It is highly recommened to train the CNN on the Peregrine HPC cluster, if the full data set is used for training. In order to copy your local data and scripts on the Peregrine HPC cluster, copy the following commands into your ``~/.bashrc`` file:

```
alias sshperigrine='ssh -X <your_P/S-number>@peregrine.hpc.rug.nl'
alias pushperegrine='rsync -avzu <path_on_your_local_machine>/dm-finder/scripts <path_on_your_local_machine>/dm-finder/cnn <your_P/S-number>@peregrine.hpc.rug.nl:/data/<your_P/S-number>/dm-finder'
alias pullperegrine='rsync -avzu <your_P/S-number>@peregrine.hpc.rug.nl:/data/<your_P/S-number>/dm-finder/cnn <path_on_your_local_machine>/dm-finder'
```
Source your ``.bashrc`` file to apply the updates via ``source ~/.bashrc``. The ``sshperigrine`` command allows you to connect to the Peregrine cluster. The ``pushperegrine`` copies your data and scripts on the Peregrine cluster. The ``pullperegrine`` command copies the output of your CNN training on your local machine. In order to run a script on the Peregrine cluster, follow the steps below:
1. login to the Peregrine cluster via ``sshperigrine``
2. Go into your working directory via ``cd /data/<your_P/S-number>``
3. Create a folder for your jobs via ``mkdir jobs``
4. Create your job, e.g. with vim via ``vim jobs/<name_of_your_job>.sh``
5. Copy the following lines into the file

```
#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=60G
#SBATCH --job-name=iact_s
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=j.j.m.aschersleben@rug.nl
#SBATCH --output=outputs/iact_images_cnn_separation.log
module restore ctapipe
python /data/p301858/dm-finder/scripts/cnn/cnn.py -m separation -i cta -na 0.5_100_TeV_fnn1_exp1 -er 0.5 100 -e 50
```

Tests can be performed with a smaller data set listed in ``dm-finder/scripts/run_lists/<particle_type>_run_list_test.csv`` on your local machine with the ``-t y`` option. The mode argument ``-m`` and the input argument ``-i`` are required in order to run the script. Other optional arguments will be discussed in the following.

##### Signal/background separation
```
python dm-finder/scripts/cnn/cnn.py -m separation -i cta
python dm-finder/scripts/cnn/cnn.py -m separation -i ps
```
The CNN can be trained for signal/background (photon/proton) separation with the CTA images ``-i cta`` or the pattern spectra ``-i ps`` as input. The pattern spectra characteristics can be specified as described in the **Create pattern spectra** section. By default, the full data set of all runs listed in ``dm-finder/scripts/run_lists/gamma_diffuse_run_list.csv`` and ``dm-finder/scripts/run_lists/proton_run_list.csv`` are considered. The energy range of the considered events can be specified with the ``-er <energy_lower> <energy_upper>`` argument. Currently, we recommend to use ``-er 0.5 100`` to consider events between 500 GeV and 100 TeV. We recommend to always specify the ``-na <name>`` argument in order to give a name to the particular experiment. The number of epochs for the CNN training can be chosen with the ``-e <number_epochs>`` argument. 

##### Energy reconstruction
```
python dm-finder/scripts/cnn/cnn.py -m energy -i cta
python dm-finder/scripts/cnn/cnn.py -m energy -i ps
```

