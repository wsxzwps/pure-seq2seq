import torch
from torch.autograd import Variable
import pickle


class Predictor(object):

    # def __init__(self, model, src_vocab, tgt_vocab):
    #     """
    #     Predictor class to evaluate for a given model.
    #     Args:
    #         model (seq2seq.models): trained model. This can be loaded from a checkpoint
    #             using `seq2seq.util.checkpoint.load`
    #         src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
    #         tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
    #     """
    #     if torch.cuda.is_available():
    #         self.model = model.cuda()
    #     else:
    #         self.model = model.cpu()
    #     self.model.eval()
    #     self.src_vocab = src_vocab
    #     self.tgt_vocab = tgt_vocab

    # def get_decoder_features(self, src_seq):
    #     src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
    #     if torch.cuda.is_available():
    #         src_id_seq = src_id_seq.cuda()

    #     with torch.no_grad():
    #         softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])

    #     return other

    # def predict(self, src_seq):
    #     """ Make prediction given `src_seq` as input.

    #     Args:
    #         src_seq (list): list of tokens in source language

    #     Returns:
    #         tgt_seq (list): list of tokens in target language as predicted
    #         by the pre-trained model
    #     """
    #     other = self.get_decoder_features(src_seq)

    #     length = other['length'][0]

    #     tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
    #     tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
    #     return tgt_seq

    # def predict_n(self, src_seq, n=1):
    #     """ Make 'n' predictions given `src_seq` as input.

    #     Args:
    #         src_seq (list): list of tokens in source language
    #         n (int): number of predicted seqs to return. If None,
    #                  it will return just one seq.

    #     Returns:
    #         tgt_seq (list): list of tokens in target language as predicted
    #                         by the pre-trained model
    #     """
    #     other = self.get_decoder_features(src_seq)

    #     result = []
    #     for x in range(0, int(n)):
    #         length = other['topk_length'][0][x]
    #         tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
    #         tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
    #         result.append(tgt_seq)

    #     return result
    def __init__(self, model, rev_vocab_path):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        with open(rev_vocab_path, 'rb') as fp:
            self.rev_vocab = pickle.load(fp, encoding='latin1')

    def rev_vocabulary(self, idx_seq):
        sentence_out = []
        for idx in idx_seq:
            sentence_out.append(self.rev_vocab[idx])
            # stop at the first __eou__
            if idx == 3:
                break

        with open('result', 'a') as f:
            f.write(' '.join(sentence_out))
            f.write('\n')

    def evaluate(self, test_data, model, loss):
        model.eval()

        loss.reset()

        batch_iterator = iter(test_data)

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths  = batch['question'], batch['qLengths']
                target_variables, target_lengths = batch['response'], batch['rLengths']
                
                if torch.cuda.is_available():
                    target_variables = target_variables.cuda()
                    input_variables = input_variables.cuda()

                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths, target_variables)
                output_d = torch.cat([decoder_outputs[i].unsqueeze(1) for i in range(len(decoder_outputs))],1)
                for i in range(output_d.shape[0]):
                    sentence = []
                    for j in range(decoder_outputs[i].shape[0]):
                        word = torch.topk(output_d[i,j,:], 1)[1]
                        sentence.append(word.item())
                    self.rev_vocab(sentence)
        
                loss.eval_batch(decoder_outputs, target_variables, target_lengths)


        return loss.get_loss()
