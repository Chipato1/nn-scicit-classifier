import numpy as np

class Layer():
    def __init__(self, n_in, n_out, hard, activation = 'Sigmoid', init = 'Xavier'):
        self.activation = activation
        # XW + b = y ; We input more than one sample per pass...
        
        if hard == True:
            #self.weights = np.array([[0.15, 0.25], [0.2, 0.30]])
            self.weights = np.array([[0.15, 0.25], [0.2, 0.30]])
            self.biases = np.array([[0.35, 0.35]])
        else:
            self.weights = np.array([[0.40, 0.5], [0.45, 0.55]])
            self.biases = np.array([[0.60, 0.6]])
        
        
        """
        self.weights = np.array([[0][]])
        self.biases = np.array([])
        
        if init == 'Xavier':
            var = np.sqrt(6.0 / (n_in + n_out))
            for i in range(n_in):
                for j in range(n_out):
                      self.weights[i,j] = np.float32(np.random.uniform(-var, var))
        print(self.weights) """
        
    def forward(self, x):
        z = x @ self.weights + self.biases

        #print(z)
        if self.activation == 'Sigmoid':
            out = 1 / (1 + np.exp(-z))
        elif self.activation == 'ReLu':
            out = np.maximum(z, 0)
        elif self.activation == 'TanH':
            out = np.tanh(z)
        else:
            out = z
        
        self.cache = (x, z)
        
        return out    
    
    def backward(self, d_out):
        inputs, z = self.cache
        weight = self.weights
        bias = self.biases
        
        if self.activation == 'Sigmoid':
            d_act = d_out * (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))
        elif self.activation == 'ReLu':
            d_act = d_out * (z > 0)
            
        elif self.activation == 'TanH':
            d_act = d_out * np.square(z)
        else:
            d_act = z
            
        #print("--------------")
    
        
        d_inputs = d_act @ weight.T
        d_weight = inputs.T @ d_act
        d_bias = d_act.sum(axis=0) 
        
        #print(d_weight.shape)
        self.d_w = d_weight
        self.d_b = d_bias
        
        return d_inputs, d_weight, d_bias
    
    def update_gd_params(self, lr):
        print("Old weights: ", self.weights)
        
        self.weights = self.weights - lr * self.d_w
        print("New weights: ", self.weights)
        self.biases = self.biases - lr * self.d_b
        print("New biases: ", self.biases)
"""
print(d_inputs)
print(d_weight)
print(d_bias)

"""