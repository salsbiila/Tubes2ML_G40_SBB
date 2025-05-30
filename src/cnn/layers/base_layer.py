class Layer:
    def __init__(self):
        self.input_tensor = None
        self.output = None
        
        self.weights = None
        self.biases = None
        
        self.d_weights = None 
        self.d_biases = None
        
        self.training_mode = True

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def initialize_parameters(self, *args, **kwargs):
        pass

    def set_training_mode(self, training: bool):
        self.training_mode = training