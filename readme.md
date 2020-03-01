# Image Quality Assessment Using Deep Learning

PyTorch implementation for image quality assessment in the [LIVE dataset](https://live.ece.utexas.edu/research/ChallengeDB/index.html). It is based on the [Neural Image Assessment (NIMA)](https://arxiv.org/abs/1709.05424) by Talebi and Milanfar.

## Getting Started
### Dataset
As mentioned before the LIVE dataset is used composed of 1162 images. Download the data and convert its labels to a .txt file with the following structure:
```
imgName, imgMOS, imgStd
```
This will be later used for generating a the necessary vectors for the groundtruth.

### Loss Function
As defined in NIMA, the Earth Mover's Disctance (EMD) is used as a ranked loss function. EMD receives two cumulative distribution functions to estimate the error. For this project the output of the network is a vector of probabilities of size 10, so the groundtruth needs to be a vector of probabilities of size 10. To do this, the .txt file with the MOS will serve the purpose to generate a truncated normal distribution using the imgMOS and imgStd in the range 1-10.

### Train a model
To train a model, the script main.py should be used. In those scripts, there is a dictionary of parameters that could be changed including architecture to use, number of epochs, batch size and folder list.