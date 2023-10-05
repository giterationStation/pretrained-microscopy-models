import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, class_weights=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.class_weights = class_weights
        self.__name__ = 'DiceBCELoss'

def forward(self, inputs, targets, smooth=1):
    inputs = F.sigmoid(inputs)
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    
    # Calculate dice loss
    intersection = (inputs_flat * targets_flat).sum()
    dice_loss = 1 - (2.*intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
    
    # Calculate BCE loss with weights
    if self.class_weights is not None:
        # Expand class_weights to match the shape of inputs and then flatten
        weights = self.class_weights[targets.long()]
        weights_flat = weights.view(-1)
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='none')
        BCE = (BCE * weights_flat).mean()
    else:
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
    
    # Combine BCE and dice loss
    Dice_BCE = self.weight * BCE + (1-self.weight) * dice_loss
    return Dice_BCE
