import numpy as np
from from_scratch.layer_simpleRNN import SimpleRNNLayer


class BidirectionalRNNLayer:
    def __init__(self, forward_weights, backward_weights, merge_mode='concat'):
        self.forward_rnn = SimpleRNNLayer(forward_weights, return_sequences=True, go_backwards=False)
        self.backward_rnn = SimpleRNNLayer(backward_weights, return_sequences=True, go_backwards=True)
        self.merge_mode = merge_mode
    
    def forward(self, x):
        forward_output = self.forward_rnn.forward(x)
        backward_output = self.backward_rnn.forward(x)

        forward_last = forward_output[:, -1, :]
        backward_last = backward_output[:, 0, :]
        
        if self.merge_mode == 'concat':
            output = np.concatenate([forward_last, backward_last], axis=1)
        elif self.merge_mode == 'sum':
            output = forward_last + backward_last
        elif self.merge_mode == 'avg':
            output = (forward_last + backward_last) / 2
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")
        
        return output