class EmbeddingLayer:
    def __init__(self, weights):
        self.weights = weights
        self.vocab_size, self.embedding_dim = weights.shape
    
    def forward(self, x):
        return self.weights[x]