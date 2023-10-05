import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, class_weights=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.class_weights = class_weights
        self.__name__ = 'DiceBCELoss'

    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute the Dice Loss
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Compute the weighted BCE
        BCE = F.binary_cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')
        
        # Combine the Dice and BCE losses
        Dice_BCE = self.weight * BCE + (1-self.weight) * dice_loss
        
        return Dice_BCE
