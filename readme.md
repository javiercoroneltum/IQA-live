# Image Quality Assessment Using Deep Learning

PyTorch implementation for image quality assessment in the [LIVE dataset](https://live.ece.utexas.edu/research/ChallengeDB/index.html). It is based on the [Neural Image Assessment (NIMA)](https://arxiv.org/abs/1709.05424) by Talebi and Milanfar.

## Getting Started

### Dataset

As mentioned before the LIVE dataset is used composed of 1162 images. Download the data and convert its labels to a .txt file with the following structure:
```imgName, imgMOS, imgStd
´´´
This will be later used for generating a the necessary vectors for the groundtruth.

#### Loss Function

As defined in NIMA, the Earth Mover's Disctance (EMD) is used as a ranked loss function. EMD receives two cumulative distribution functions to estimate the error. For this project the output of the network is a vector of probabilities of size 10, so the groundtruth needs to be a vector of probabilities of size 10. To do this, the .txt file with the MOS will serve the purpose to generate a truncated normal distribution using the imgMOS and imgStd in the range 1-10.

### Train a model

To train a model, the script main.py should be used. In those scripts, there is a dictionary of parameters that could be changed including architecture to use, number of epochs, batch size and folder list.



## Folder structure ith detailed description
opus
├── data
│   ├── Flipped (Folder with the correctly aligned data)
│   ├── mice (Mice data from Lund study)
│   ├── Part1 (Folder with studies part1)
│   └── Part3Cropped (Part3 cropped to from the skin line)
├── notebooks
│   ├── ClassificationCIFAR.ipynb (Classification experiments on CIFAR)
│   ├── CNN Visualization AlexNet-Imagenet.ipynb (GradCAM for AlexNet with Natural Images)
│   ├── CNN Visualization AlexNetUS.ipynb (GradCAM for AlexNet US)
│   ├── CNN Visualization DuAlexNet.ipynb (GradCAM for AlexNet OPUS)
│   ├── Cropped Images.ipynb (To crop the images from part1 or part3)
│   ├── Cross-Validation Options.ipynb (Notebook to check data splitting and cross-validation)
│   ├── Data Augmentation.ipynb (Augment data from specific folders)
│   ├── Dataloader Viewer.ipynb (Visualization of the dataloaders)
│   ├── Dataloader Viewer Mice.ipynb (Visualization of the dataloaders for mice)
│   ├── DenseNet Grad-CAM.ipynb (GradCAM for DenseNet in Natural Images)
│   ├── DuAlexNet-DenseNet Integration.ipynb (AlexNet-Densenet)
│   ├── OPUS (Dual)DenseNet Grad-CAM.ipynb (GradCAM for OPUS DenseNet)
│   ├── Transfer learning.ipynb (Notebooks with experiments for loeading trained models and create new ones)
│   └── HeatMaps (Folder containing heatmaps for DenseNet)
├── dataOld (Data containing all the images, bad labeled inlcuded)
├── excludedData (Data after augmentation, excluded as seen not useful)
│
├── model
│   ├── classifier.py (Defines optimizer and loss function)
│   ├── densenet.py (Original densenet script)
│   ├── dualmoduleCIFAR.py (Dual-DenseNet architectufre for CIFAR, from densenet)
│   ├── dualmodule.py (Dual-DenseNet architectufre, from densenet)
│   ├── models.py (Other listed architectures and architecture selector)
│   ├── visualizerOPUS.py (GradCAM script for OPUS)
│   └── visualizer.py (GradCAM sccript for natural images)
├── utils
│   ├── data_loader.py (Script that defines the data loader)
│   └── helper_functions.py (Data handling, logers and other functions)
│
├── inference.py (Loads a model and data from folders to do inference only)
├── mainCV.py (Performs a crossvalidation based on splitted files)
├── mainLOO.py (Performs leave one out training)
├── main.py (Performs kfold cross-validation or training with k repetitions for a specific data split)
│ 
├── train.py (Script with the main training pipeline)
├── evaluate.py (Script for validation loop)
└── test.py (Script for test loop)