# dm-finder
## Installation
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

### CTA data download
Use ``mdkir -p dm-finder/data/gamma/event_files`` to create the ``event_files`` directory and download the CTA data with ``DiRAC`` into the ``event_files`` directory. 
