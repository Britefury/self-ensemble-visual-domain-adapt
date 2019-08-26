# Self-ensembling for visual domain adaptation (small images)

Implementation of the paper [Self-ensembling for visual domain adaptation](https://arxiv.org/abs/1706.05208),
accepted as a poster at ICLR 2018.

For small image datasets including MNIST, USPS, SVHN, CIFAR-10, STL, GTSRB, etc.

For the VisDA experiments go to
[https://github.com/Britefury/self-ensemble-visual-domain-adapt-photo/](https://github.com/Britefury/self-ensemble-visual-domain-adapt-photo/).


## Installation

You will need:

- Python 3.6 (Anaconda Python recommended)
- OpenCV with Python bindings
- PyTorch

First, install OpenCV and PyTorch as `pip` may have trouble with these.

### OpenCV with Python bindings

On Linux, install using `conda`:

```> conda install opencv```

On Windows, go to [https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and
download the OpenCV wheel file and install with:

```> pip install <path_of_opencv_file>```

### PyTorch

For installation instructions head over to the [PyTorch](https://pytorch.org/) website.

### The rest

Use pip like so:

```> pip install -r requirements.txt```

## Usage

Domain adaptation experiments are run via the `experiment_domainadapt_meanteacher.py` Python program.

The experiments in our paper can be re-created by running the `batch_search_exp.sh` shell script like so:

```bash batch_search_exp.sh <GPU> <RUN>```

Where `<GPU>` is a string identifying the GPU to use (e.g. `cuda:0`) and <RUN> enumerates the experiment number so that
you can keep logs of multiple repeated runs separate, e.g.:

```bash batch_search_exp.sh cuda:0 01```

Will run on GPU 0 and will generate log files with names suffixed with `run01`.

To re-create the supervised baseline experiments:

``` bash batch_search_exp_sup.sh <GPU> <RUN```

Please see the contents of the shell scripts to see the command line options used to control the
experiments.


## Syn-Digits, GTSTB and Syn-Signs datasets

You will need to download the Syn-Digits, GTSRB and Syn-signs datasets. After this you will need to create
the file `domain_datasets.cfg` to tell the software where to find them.

The following assumes that you have a directory called `data` in which you will store these three datasets.

### Syn-digits

Download Syn-digits from [http://yaroslav.ganin.net](http://yaroslav.ganin.net), on which you will find a Google Drive
link to a file called `SynthDigits.zip`. Create a directory call `syndigits` within `data` and unzip `SynthDigits.zip`
within it.

### GTSRB

Download GTSRB from [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
and get the training 'Images and annotations' (`GTSRB_Final_Training_Images.zip`), Test 'images and annotations' (`GTSRB_Final_Test_Images.zip`)
and the test 'extended annotations including class IDs' (`GTSRB_Final_Test_GT.zip`).

Unzip the three files within the `data` directory. You should end up with the following directory structure:

```
GTSRB/
GTSRB/Final_Training/
GTSRB/Final_Training/Images/   -- training set images
GTSRB/Final_Training/Images/00000/   -- one directory for each class, contains image files
GTSRB/Final_Training/Images/00001/
...
GTSRB/Final_Training/Images/00042/
GTSRB/Final_Test/
GTSRB/Final_Test/Images/   -- test set images
GTSRB/GT-final_test.csv   -- test set ground truths
GTSRB/Readme-Images.txt
GTSRB/Readme-Images-Final-test.txt
``` 

#### Prepare GTSRB

Convert GTSRB to the required format using:

```sh
> python prepare_gtsrb.py
```

### Syn-signs
Download Syn-signs from [http://graphics.cs.msu.ru/en/node/1337/](http://graphics.cs.msu.ru/en/node/1337/).
You should get a file called `synthetic_data.zip`. Create a directory called `synsigns` within data and unzip
`synthetic_data.zip` within `data/synsigns` to get the following:

```
synthetic_data/
synthetic_data/train/   -- contains the images as PNGs
synthetic_data/train_labelling.txt   -- ground truths
```

#### Prepare Syn-signs

Convert Syn-signs to the required format using:

```sh
> python prepare_synsigns.py
```


### Create `domain_datasets.cfg`

Create the configuration file `domain_datasets.cfg` within the same directory as the experiment scripts.
Put the following into it (change the paths if they are different):

```cfg
[paths]
syn_digits=data/syndigits
gtsrb=data/GTSRB
syn_signs=data/synsigns
```




