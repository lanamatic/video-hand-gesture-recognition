"""
LSTM / BiLSTM / GRU classifier for landmark sequences.

rnn_type='lstm' + bidirectional=False -> LSTM
rnn_type='lstm' + bidirectional=True  -> BiLSTM
rnn_type='gru'                        -> GRU
"""

import torch
import torch.nn as nn

NUM_CLASSES = 13
NUM_FEATURES = 63

class GestureRNN(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, hidden_size=128, num_layers=2,
                 num_classes=NUM_CLASSES, rnn_type='lstm', bidirectional=True,
                 dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(packed)

        h_n = hidden[0] if self.rnn_type == 'lstm' else hidden   

        if self.bidirectional:     
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]

        return self.head(last)      
