'''Helper functions.
'''

import torch


def calculate_accuracy(predicted, target):
    '''Calculates the accuracy of the prediction.
    '''

    num_data = target.size()[0]
    predicted = torch.argmax(predicted, dim=1)
    correct_pred = torch.sum(predicted == target)

    accuracy = correct_pred*(100/num_data)

    return accuracy.item()
