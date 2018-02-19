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

On Linux:

```> conda install pytorch torchvision -c pytorch```

On Windows:

```> conda install -c peterjc123 pytorch cuda90```

### The rest

Use pip like so:

```> pip install -r requirements.txt```

## Usage

Domain adaptation experiments are run via the `experiment_domainadapt_meanteacher.py` Python program.

The experiments in our paper can be re-created by running the `batch_search_exp.sh` shell script like so:

```bash batch_search_exp.sh <GPU> <RUN>```

Where `<GPU>` is an integer identifying the GPU to use and <RUN> enumerates the experiment number so that
you can keep logs of multiple repeated runs separate, e.g.:

```bash batch_search_exp.sh 0 01```

Will run on GPU 0 and will generate log files with names suffixed with `run01`.

To re-create the supervised baseline experiments:

``` bash batch_search_exp_sup.sh <GPU> <RUN```

Please see the contents of the shell scripts to see the command line options used to control the
experiments.