from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss.loss import Criterion
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint

import math
class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=Criterion, batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, target_lengths, model, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Get loss
        loss.reset()        
        # for step, step_output in enumerate(decoder_outputs):
        #     batch_size = target_variable.size(0)
        #     loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
        loss.eval_batch(decoder_outputs, target_variable, target_lengths)
        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        # batch_iterator = torchtext.data.BucketIterator(
        #     dataset=data, batch_size=self.batch_size,
        #     sort=False, sort_within_batch=True,
        #     sort_key=lambda x: len(x.src),
        #     device=device, repeat=False)
        
        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))
            ld = iter(data)
            model.train(True)
            for batch in ld:
                step += 1
                step_elapsed += 1
                inputs = batch
                
                input_variables, input_lengths = inputs['question'], inputs['qLengths']
                target_variables = inputs['response']
                target_lengths = inputs['rLengths']

                if torch.cuda.is_available():
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                loss = self._train_batch(input_variables, input_lengths, target_variables, target_lengths, model, teacher_forcing_ratio)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train loss: %.4f, Perplexity: %.4f' % (
                        step / total_steps * 100,
                        print_loss_avg, math.exp(print_loss_avg))
                    log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer': self.optimizer,
                        'loss': self.loss
                        }, self.expt_dir+'/checkpoint')

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train loss: %.4f, Perplexity: %.4f" % (epoch, epoch_loss_avg, math.exp(epoch_loss_avg))
            if dev_data is not None:
                dev_loss = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev loss: %.4f, Perplexity: %.4f" % (dev_loss, math.exp(dev_loss))
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters(), lr=0.0002), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
