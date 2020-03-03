from utils import helper_functions as hf
from torch.autograd import Variable
import numpy as np, os
import random, logging
from tqdm import tqdm
import torch
import time


def run_session(model, trainData, valData, optimizer, lossFn, params):
    """ Performs training and validation using a learning rate scheduler. """

    # Reduce on Plateau Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, verbose=True)
    corr = 0

    with tqdm(total=params['numEpochs'], ascii=True) as (t):
        for epoch in range(params['numEpochs']):
            t.set_description('Epoch %i' % epoch)

            trainLoss = train(model, optimizer, lossFn, trainData, params)
            valLoss, metrics = evaluate(model, lossFn, valData, params, epoch)
            
            scheduler.step(valLoss, epoch)
            logging.info("Epoch {}: TrainLoss:{:05.4f}, ValLoss:{:05.4f}, Spearman:{:05.4f}, Pearson:{:05.4f}".format(epoch, trainLoss, valLoss, metrics["spearman"][0], metrics["pearson"][0]))

            hf.tensorboard_logger(params['modelDir'], epoch=epoch, trainLoss=trainLoss,  valLoss=valLoss,
                                                    valPearson=metrics['pearson'][0], valSpearman=metrics['spearman'][0])

            # Chech if save the model based on correlation value
            if corr < (metrics["spearman"][0]+metrics["pearson"][0])/2:#
                corr = (metrics["spearman"][0]+metrics["pearson"][0])/2
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            params['modelDir']+'.pt')

            t.update(1)
    print('\n')
    return #metricsVal


def train(model, optimizer, lossFn, dataLoader, params):
    """ Computes a training step for one epoch (one full pass over the training set). 
    """
    
    model.train()
    lossValue = hf.RunningAverage()

    with tqdm(total=len(dataLoader), ascii=True) as (t):
        t.set_description('Training')
        for i, (trainBatch, labelBatch) in enumerate(dataLoader):
            trainBatch, labelBatch = trainBatch.cuda(async=True), labelBatch.cuda(async=True)
            trainBatch, labelBatch = Variable(trainBatch), Variable(labelBatch)
            outputBatch = model(trainBatch)
            loss = lossFn(outputBatch.float(), labelBatch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossValue.update(loss.item())
            t.set_postfix(loss=('{:05.3f}').format(lossValue()))
            t.update()
    #metrics = hf.get_metrics(paths, probs, mosPred, mosTrue, params, train=True)
    
    return lossValue.avg#, metrics)


def evaluate(model, lossFn, dataLoader, params, epoch=-1):
    """ Computes validation for one epoch (one full pass over the validation set). 
    """

    # Set model to evaluation mode
    model.eval()

    # Initialize probabilities, labels, paths and running average for the loss, later used for update
    lossValue = hf.RunningAverage()
    paths = []
    probs = np.zeros((1, 10))
    mosPred = []; mosTrue = []

    with tqdm(total=len(dataLoader), ascii=True) as t:
        if epoch == -1: t.set_description('Test')
        else: t.set_description('Validation')

        for i, (valBatch, labelBatch, path) in enumerate(dataLoader):

            valBatch, labelBatch = valBatch.cuda(async=True), labelBatch.cuda(async=True)
            valBatch, labelBatch = Variable(valBatch), Variable(labelBatch)
            outputBatch = model(valBatch)
            loss = lossFn(outputBatch.float(), labelBatch.float())

            paths = np.concatenate((paths, path))
            probs = np.concatenate((probs, outputBatch.detach().cpu().numpy()))
            mosTrue = np.concatenate((mosTrue, (labelBatch.detach().cpu().numpy()*np.array([1,2,3,4,5,6,7,8,9,10])).sum(axis=1)))
            mosPred = np.concatenate((mosPred, (outputBatch.detach().cpu().numpy()*np.array([1,2,3,4,5,6,7,8,9,10])).sum(axis=1)))

            # Update the average loss
            lossValue.update(loss.item())
            
            t.set_postfix(loss='{:05.3f}'.format(lossValue()))
            t.update()

    metrics = hf.get_metrics(paths, probs, mosPred, mosTrue, params)
    
    if epoch == -1:#True:#epoch == (params["numEpochs"]-1):#params["bestAUC"] < metrics['AUC']:#
        if(os.path.isdir(params['modelDir'])==False): os.makedirs(params['modelDir'])
        #logging.info("Generating and saving validation")
        hf.generate_figures(mosTrue, mosPred, params)
        logging.info("Test: Spearman:{:05.4f}, Pearson:{:05.4f}".format(metrics["spearman"][0], metrics["pearson"][0]))
        with open(os.path.join(params['modelDir'], 'TestOutputs.log'), 'w') as (file_handler):
            file_handler.write(('{}\n').format(('\n').join(map(str, sorted(metrics['results'])))))  

    return lossValue.avg, metrics    