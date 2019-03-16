import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import torch.optim as optim

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss.loss import Criterion
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.loader.loader import CustomDataset, LoaderHandler

import numpy as np

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--test_path', action='store', dest='test_path',
                    help='Path to test data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')


opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare dataset
max_len = 100


wordDict = 'AuxData/wordDict'
data_paths = {
    'train':'Data/train',
    'dev':'Data/dev',
    'test':'Data/test'
}
loader = LoaderHandler(wordDict, data_paths, 32)
train = loader.ldTrain
dev = loader.ldDev
test = loader.ldTestEval

# Initialize model
hidden_size=100
bidirectional = True

# hard coded some arguments for now
embedding_path = 'AuxData/word2vec.npy'
rev_vocab_path = 'AuxData/rev_vocab'
embedding = torch.FloatTensor(np.load(embedding_path))
vocab_size = len(embedding)
sos_id = 2
eos_id = 3

encoder = EncoderRNN(vocab_size, max_len, hidden_size,
                        bidirectional=bidirectional, variable_lengths=True, embedding=embedding)
decoder = DecoderRNN(vocab_size, max_len, hidden_size * 2 if bidirectional else hidden_size,
                        dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                        eos_id=eos_id, sos_id=sos_id)

seq2seq = Seq2seq(encoder, decoder)
optimizer = optim.Adam(seq2seq.parameters(), lr=0.0002)


if torch.cuda.is_available():
    seq2seq = seq2seq.cuda()

for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)


if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, 'checkpoint')))
    checkpoint_path = os.path.join(opt.expt_dir, 'checkpoint')
    checkpoint = torch.load(checkpoint_path)

    seq2seq.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

else:
    loss = Criterion()

opti = Optimizer(optimizer, max_grad_norm=5)


t = SupervisedTrainer(loss=loss, batch_size=32,
                        checkpoint_every=1000,
                        print_every=10, expt_dir=opt.expt_dir)

seq2seq = t.train(seq2seq, train,
                    num_epochs=1, dev_data=dev,
                    optimizer=opti,
                    teacher_forcing_ratio=0.5,
                    resume=opt.resume)

predictor = Predictor(seq2seq,rev_vocab_path)
predictor.evaluate(test, seq2seq, loss)


