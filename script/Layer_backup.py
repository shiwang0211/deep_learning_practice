
class Layer:
    def __init__(self, activation_function, num_neurons, batch_norm = False, dropout_p = 1.0):
        self.dim = num_neurons
        self.activation = activation_function
        self.batch_norm = batch_norm
        if batch_norm:
            self.batchnorm = BatchNorm()
        self.isfirst = False
        self.islast = False
        self.before = None
        self.p = dropout_p

    def set_first_layer(self, input):
        self.isfirst = True
        self.X = input
        
    def set_last_layer(self, y):
        self.islast = True
        self.y = y
    
    def initialize_Wb(self):
        if self.isfirst:
            before_dim = self.X.shape[1]
        else:
            before_dim = self.before.dim
        self.W = np.random.randn(before_dim, self.dim) / np.sqrt(before_dim) # see notes above
        self.b = np.random.randn(self.dim).reshape(1, self.dim) # see notes above
        self.gamma, self.beta = (1., 0.)

    def forward_propagation(self):
        if not self.isfirst:
            self.X = self.before.Z
            
        self.mask = np.random.rand(*self.W.shape) < self.p / self.p
        self.W *= self.mask
        self.WX = Mul().forward( self.W, self.X )
        self.S = Add().forward( self.WX, self.b)
        if self.batch_norm:
            self.SZ = self.batchnorm.forward( self.S, self.gamma, self.beta, eps = 0)
        else:
            self.SZ = self.S
        self.Z = self.activation.forward(self.SZ)
            
    def backward_propagation(self):
        if self.islast:
            self.dZ = self.y
        
        self.dSZ = self.activation.backward(self.SZ, self.dZ)
        if self.batch_norm:
            self.dS, self.dgamma, self.dbeta = self.batchnorm.backward(self.dSZ)
        else:
            self.dS = self.dSZ
            self.dgamma, self.dbeta = 0,0
        self.db, self.dWX = Add().backward(self.WX, self.b, self.dS)
        self.dW, self.dX = Mul().backward(self.W, self.X, self.dWX)
        self.dW *= self.mask
        
        if not self.isfirst:
            self.before.dZ = self.dX
    
    def update_weight(self, learning_rate, lambda_ , method):
        
        # Create variable list
        self.weights = [self.W, self.b, self.gamma, self.beta]
        self.ds = [self.dW, self.db, self.dgamma, self.dbeta]

        # First Time
        if not hasattr(self, 'updates'):      
            self.updates = []
            for weight in self.weights:
                self.updates.append(WeightUpdate(weight))
        
        # Calculate update for each iteration
        new_weights = []
        for weight_update, d in zip(self.updates, self.ds):
            new_weights.append(weight_update.Update(d, learning_rate, lambda_, method))
        
        # Update weights
        self.W, self.b, self.gamma, self.beta = new_weights
        
        # Vanilla version
        #self.W += -learning_rate * self.dW + (- lambda_ * self.W)
        #self.b += -learning_rate * self.db
        #self.gamma += -learning_rate * self.dgamma
        #self.beta  += -learning_rate * self.dbeta
            
    # Only for softmax layer
    def calculate_loss(self):
        loss = self.activation.forward_loss(self.Z, self.y)
        return loss
            
    def predict(self):
        return self.activation.predict(self.Z)
    
    def calculate_acc(self): 
        pred = self.predict()
        return sum( pred == self.y ) / len(self.y)   