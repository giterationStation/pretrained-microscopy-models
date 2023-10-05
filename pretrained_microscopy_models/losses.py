import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.__name__ = 'DiceBCELoss'

    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        # Number of classes
        num_classes = inputs.shape[1]
        
        # List to store per-class losses
        per_class_losses = []
        
        for cls in range(num_classes):
            input_cls = inputs[:, cls, ...].contiguous().view(-1)
            target_cls = targets[:, cls, ...].contiguous().view(-1)
            
            intersection = (input_cls * target_cls).sum()                            
            dice_loss = 1 - (2.*intersection + smooth) / (input_cls.sum() + target_cls.sum() + smooth)
            BCE = F.binary_cross_entropy(input_cls, target_cls, reduction='mean')
            Dice_BCE = self.weight * BCE + (1-self.weight) * dice_loss
            
            per_class_losses.append(Dice_BCE)
        
        return per_class_losses