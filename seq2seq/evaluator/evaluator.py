from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()

        batch_iterator = iter(data)

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths  = batch['question'], batch['qLengths']
                target_variables, target_lengths = batch['response'], batch['rLengths']
                
                if torch.cuda.is_available():
                    target_variables = target_variables.cuda()
                    input_variables = input_variables.cuda()

                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths, target_variables)
        
                loss.eval_batch(decoder_outputs, target_variables, target_lengths)


        return loss.get_loss()
