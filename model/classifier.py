import torch
import logging
import torch.optim as optim
import torch.nn as nn


def __init__(self, net, params):
    return


class EMDLoss(nn.Module):
    """ Defines the Earth Mover's Distance loss as in NIMA """

    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, pTarget: torch.Tensor, pEstimate: torch.Tensor):
        assert pTarget.shape == pEstimate.shape
        # Cumulative Distribution Function
        cdfTarget = torch.cumsum(pTarget, dim=1)
        cdfEstimate = torch.cumsum(pEstimate, dim=1)
        cdfDiff = cdfTarget - cdfEstimate
        sampleEMD = torch.sqrt(torch.mean(torch.pow(torch.abs(cdfDiff), 2)))
        
        return sampleEMD.mean()


class MSELoss(nn.Module):
    """ Defines the Mean Squared Error Loss """

    def __init__(self):
        super(MSELoss, self).__init__()
        self.lossFunction = nn.MSELoss(reduction='mean')

    def forward(self, pTarget: torch.Tensor, pEstimate: torch.Tensor):
        assert pTarget.shape == pEstimate.shape
        lossValue = self.lossFunction(pTarget.float(), pEstimate.float())

        return lossValue


class CombinedLoss(nn.Module):
    """ Combines EMD and MSE as a single loss function, weighting can be modified as additional parameter"""

    def __init__(self, weights=[1,1]):
        super(CombinedLoss, self).__init__()
        self.MSELoss = MSELoss()
        self.EMDLoss = EMDLoss()
        self.weights = weights

    def forward(self, pTarget: torch.Tensor, pEstimate: torch.Tensor):
        assert pTarget.shape == pEstimate.shape
        MSEValue = self.MSELoss(pTarget,pEstimate)
        EMDValue = self.EMDLoss(pTarget,pEstimate)
        lossValue = (self.weights[0]*EMDValue) + (self.weights[1]*MSEValue)
        
        return lossValue


def get_optimizer(model, params):
    """ Defines the optimizer to use during the training procedure """

    if params["optimizerName"] == "Adam":
        logging.info('Using ' + params["optimizerName"]  + ' as optimizer')

        return optim.Adam(model.parameters(), lr=params["lr"])#, weight_decay=1e-4)

    elif params["optimizerName"] == "SGD":
        logging.info('Using ' + params["optimizerName"]  + ' as optimizer')

        return optim.SGD(model.parameters(), lr=params["lr"])#, weight_decay=1e-4, nesterov=True, momentum=0.9) 

    elif params["optimizerName"] == "RMSprop":
        logging.info('Using ' + params["optimizerName"]  + ' as optimizer')

        return optim.RMSprop(model.parameters(), lr=params["lr"], momentum=0.6)       