import torch
from torch.nn.utils.rnn import pad_sequence

PAD_token = ("</pad>", 0)
SOS_token = ("</sos>", 1)
EOS_token = ("</eos>", 2)


def seq_to_one_hot(seq, depth: int):
    """
    Encode elements of sequences as one hot vectors.
    """
    return torch.eye(depth)[seq]


def collate_fn(batch):
    (x, y) = zip(*batch)

    x = torch.stack(x)
    x_len = torch.tensor(x.shape[1]).expand(x.shape[0]).clone()

    y = torch.stack(y)
    if len(y.shape) == 1:
        y_len = torch.tensor(0).expand(y.shape[0]).clone()
    else:
        y_len = torch.tensor(y.shape[1]).expand(y.shape[0]).clone()

    return x, y, x_len, y_len


def collate_fn_pad(batch, padding_value=0.0):
    (x, y) = zip(*batch)

    x_len = torch.LongTensor(list(map(len, x)))
    x = pad_sequence(x, padding_value=padding_value, batch_first=True)
    x_len, per_idx = x_len.sort(0, descending=True)
    x = x[per_idx]

    if y[0].shape == torch.Size([]):
        y = torch.stack(y)
        y_len = torch.tensor(0).expand(y.shape[0]).clone()
    else:
        y_len = torch.LongTensor(list(map(len, y)))
        y = pad_sequence(y, padding_value=padding_value, batch_first=True)

    y = y[per_idx]
    y_len = y_len[per_idx]

    return x, y, x_len, y_len


class Dictionary:
    def __init__(self, datasets: list):
        self.vocab = set([j for i in datasets for j in i])
        self.vocab.add(PAD_token[0])

        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = [w for i, w in enumerate(self.vocab)]

        self.swap_position(PAD_token[0], PAD_token[1])

        if SOS_token[0] in self.word2id:
            self.swap_position(SOS_token[0], SOS_token[1])

        if EOS_token[0] in self.word2id:
            self.swap_position(EOS_token[0], EOS_token[1])

    def swap_position(self, word, pos):
        self.word2id[self.id2word[pos]] = self.word2id[word]
        self.id2word[self.word2id[word]] = self.id2word[pos]

        self.id2word[pos] = word
        self.word2id[word] = pos

    def __len__(self):
        return len(self.id2word)
