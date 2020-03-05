import os, sys
sys.path.insert(0, './')
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from utils import helper_functions as hf
from model import models, classifier
from utils import data_loader as dl
from pipeline import run_session, evaluate
import numpy as np
import os, sys
import logging
import torch

# Script to run the experiments with a datasplit that yields good test results

# Specify the paraeters and configurations for the runs
params = {}
params["Comments"] = "DenseNet Natural Images LIVE dataset"
params["runName"] = "live"
params["runsNumber"] = 1
params["architecture"] = "MobileNet"
params["optimizerName"] = "Adam"#"SGD"#"RMSprop"
params["lr"] = 1e-4
params["numEpochs"] = 4
params["batchSize"] = 15
params["modelDir"] = hf.get_name(params)#os.path.join("experiments", params["modelName"])
params["mainModelDir"] = hf.get_name(params)#os.path.join("experiments", params["modelName"])
params["imgSize"] = 244#32#224#299#
# Define Data and Labels folder
params["dataFolder"] = "data/Images/"
params["scoresFile"] = "data/labels.txt"

# List all images in data folder and create dictionary of scores
imgsList = hf.get_image_paths(params["dataFolder"], natural=True)
scoresDicts = hf.get_scores_dict(params["scoresFile"])

# Create model folder and set the logger
if not os.path.exists(params["modelDir"]): os.makedirs(params["modelDir"])
hf.set_logger(os.path.join(params["modelDir"],'train.log'))
logging.info('Parameters for run: %s', "{" + "\n".join("{}: {}".format(k, v) for k, v in params.items()) + "}")

# Number of runs loop
for runId in range(0,params["runsNumber"]):
    logging.info("####################################################################")
    # Set folder for current run
    params["modelDir"] = os.path.join(params["mainModelDir"], "run_"+str(runId+1))
    randState= int(np.random.rand(1)*10*(runId+1))
    logging.info("Random state set to "+str(randState))

    # Data splitting for training, validation and testing
    trainVal, testPaths = train_test_split(imgsList, test_size=0.2, random_state=randState, shuffle=True)
    trainPaths, valPaths = train_test_split(trainVal, test_size=0.2, random_state=randState, shuffle=True)

    logging.info("Data for Testing: {}".format(str(testPaths)))
    logging.info("Data for Validation: {}".format(str(valPaths)))
    logging.info("Data for Training: {}".format(str(trainPaths)))
    
    # Load training data
    logging.info("Loading the datasets for run_"+str(runId+1))
    logging.info("Using {} images for training".format(len(trainPaths)))
    trainingSet = dl.basicDataset(trainPaths, scoresDicts, params)
    trainLoader = torch.utils.data.DataLoader(trainingSet, batch_size=params["batchSize"], shuffle=True, num_workers=0)

    # Load validation data
    logging.info("Using {} images for validation".format(len(valPaths)))
    validationSet = dl.basicDataset(valPaths, scoresDicts, params, val=True)
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=params["batchSize"], shuffle=True, num_workers=0)

    # Load test data
    logging.info("Using {} images for training".format(len(testPaths)))
    testSet = dl.basicDataset(testPaths, scoresDicts, params, val=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=params["batchSize"], shuffle=True, num_workers=0)

    # Define model, optimizer and fetch loss function
    model = models.get_architecture(params)
    model = model().cuda() # Move to gpu
    optimizer = classifier.get_optimizer(model, params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = classifier.CombinedLoss(weights=[2,0.5]).to(device)
    
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params["numEpochs"]))
    run_session(model, trainLoader, validationLoader, optimizer, loss, params)

    # Test the model
    metrics = evaluate(model, loss, testLoader, params, -1)
    
    # Update the metrics
    # logging.info("Accuracy: {}".format(metrics["Accuracy"]))
    
    logging.info("Run " + str(runId+1) + " done")

    # Clear variables and gpu
    del model, optimizer, loss, trainLoader, validationLoader, testLoader, metrics
    torch.cuda.empty_cache()

logging.info("####################################################################")
logging.info("Cross-validation finished, reporting metrics")