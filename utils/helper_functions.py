from tensorboardX import SummaryWriter
import scipy
import numpy as np
import logging
import os, sys
import re

def get_scores_dict(scoresFile):
    """ Creates a dictionary with the image names and its listed scores """
    
    scores = {}

    f = open(scoresFile, "r")
    for line in f:
        fields = line.split(",")
        scores[fields[0]] = np.array(fields[1:]).astype(np.float)
    
    return scores


def get_image_paths(folderPath, natural=False):
    """ List all OP images in folder """
    
    allImages = sorted(os.listdir(folderPath))
    allImages[:] = [os.path.join(folderPath, item) for item in allImages]

    return allImages


def get_paths_if_study(pathsList, studiesList):
    """ From a list of paths return only those included in a second list"""

    imgPaths = []
    imgPaths[:] = [item for item in pathsList if any(sub in item for sub in studiesList)]

    return imgPaths


def get_metrics(paths, probs, mosPred, mosTrue, params, train=False):
    """ Produce the metrics given the outputs of the network
    """
    metrics = {}
    metrics["spearman"] = scipy.stats.pearsonr(mosTrue, mosPred)
    metrics["pearson"] = scipy.stats.spearmanr(mosTrue, mosPred)
    metrics["results"] = all_outputs(paths, probs, mosPred, mosTrue)

    return metrics


def all_outputs(paths, probs, mosPred, mosTrue):
    """ Given the cumulative outputs of the network for validation
        return the path of the image with the results with structure:
        path, raw output, softmax(output), classification
    """

    results = []
    for item in range(0,len(paths)):
        results.append([paths[item], probs[item+1], mosPred[item], mosTrue[item]])

    return results


def set_logger(log_path):
    """Set the logger to log info in terminal and file`log_path.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
        self.avg = self.total/float(self.steps)
    
    def __call__(self):
        return self.total/float(self.steps)        


def tensorboard_logger(modelDir, *args, **kwargs):
    """ Callback intended to be executed at each epoch of the training which goal is to add valuable
        information to the tensorboard logs such as the losses and accuracies
    Args:
        description (str): Description used for folder path
    """
    epoch = kwargs['epoch']+1
    
    # Write files in folder for TensorBoardX for training
    if(os.path.isdir(modelDir)==False):
        os.makedirs(modelDir)
    writer = SummaryWriter(modelDir) # this is a folder
    writer.add_scalars('Loss', {'Training': kwargs['trainLoss'], 'Validation': kwargs['valLoss']}, epoch)
    writer.add_scalars('Correlation', {'Spearman': kwargs['valSpearman'], 'Pearson': kwargs['valPearson']}, epoch)
    
    writer.close()


def get_name(params):
    """ Generate the name given to the folder/experiment where to save the results.
        Takes into account the given parameters for the run
    """

    folderDescription = params["runName"] + "/" + params['architecture'] + '_B'+str(params['batchSize']) + \
                '_lr' + str(str(format(params['lr'], ".1e"))) + \
                '_' + str(params['optimizerName'])
    folderName = os.path.join("experiments", folderDescription)
    if not os.path.exists(folderName): os.makedirs(folderName)
    runsInFolder = len(next(os.walk(folderName))[1])
    runNumber = "run_" + str(runsInFolder+1)
    folderName = os.path.join(folderName, runNumber)

    return folderName
