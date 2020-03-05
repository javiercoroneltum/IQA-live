import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
import scipy.stats as stats
from skimage import io
import numpy as np
import os, sys
import torch


class basicDataset(Dataset):
    """ Dataset generator for training with given parameters """
    
    def __init__(self, imgPaths, scoresDict, params, val=False):
        """Initialization"""
        self.imgPaths = imgPaths
        self.scoresDict = scoresDict
        self.params = params
        self.val = val
        
    def __getitem__(self, index):
        """Generates one sample of data"""
        imgPath = self.imgPaths[index]
        scoresDict = self.scoresDict
        val = self.val
        
        # For label
        imgName = imgPath[imgPath.find("Images")+7:]
        [mu, sigma] = scoresDict[imgName]
        y = stats.truncnorm.rvs((0 - mu) / sigma, (100 - mu) / sigma, loc=mu, scale=sigma, size=100).round().astype(int)
        scoreHist = np.histogram(y/10, bins=[1,2,3,4,5,6,7,8,9,10,11])[0]
        normScoreHist =  np.around(scoreHist,3)/scoreHist.sum()
        score = torch.tensor(normScoreHist)

        # Load image
        img = io.imread(imgPath)
        # Transformations
        img = self.transformations(img, self.params)        

        if val:
            return img, score, imgName
        else:
            return img, score

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.imgPaths)

    def transformations(self, img, config):

        # Convert to PIL image
        img = TF.to_pil_image(img)
        # Resize images
        img = TF.resize(img, size=(config["imgSize"], config["imgSize"]))
        # Rotate images
        if np.random.choice([1,0]): img = TF.hflip(img)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[img.mean()], std=[img.std()])

        return img