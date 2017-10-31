# Self-ensembling for visual domain adaptation (small images)

Implementation of the paper 'Self-ensembling for visual domain adaptation', submitted to ICLR 2018.

For small image datasets including MNIST, USPS, SVHN, CIFAR-10, STL, GTSRB, etc.

## Installation

You will need:

- Python 3.6 (Anaconda Python recommended)
- OpenCV with Python bindings
- PyTorch

First, install OpenCV and PyTorch as `pip` may have trouble with these.

### OpenCV with Python bindings

On Linux, install using `conda`:

```> conda install -c menpo opencv```

On Windows, go NOTE-TO-SELF <url here> and download the OpenCV wheel file and install with:

```> pip install <path_of_opencv_file>```

### PyTorch

On Linux:

```> conda install -c soumith pytorch```

On Windows:

```> conda install -c peterjc123 pytorch```

### The rest

Use pip like so:

```> pip install -r requirements.txt```
