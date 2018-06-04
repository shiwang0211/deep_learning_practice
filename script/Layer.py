from script.Propagation import *
class BaseLayer:
    def __init__(self):
        self.isfirst = False
        self.islast = False
        self.before = None
        self.after = None
        self.gamma, self.beta = 1,0
        self.dgamma, self.dbeta = 0,0

    def set_first_layer(self, input):
        self.isfirst = True
        self.X = input
        
    def set_last_layer(self, y):
        self.islast = True
        self.y = y
    
    def initialize_Wb(self):
        raise NotImplementedError()

    def forward_propagation(self):
        raise NotImplementedError()
            
    def backward_propagation(self):
        raise NotImplementedError()
    
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
class FC(BaseLayer):
    def __init__(self, activation_function, num_neurons, batch_norm = False, dropout_p = 1.0): # p = 1 means no dropout
        super().__init__()
        self.dim = num_neurons
        self.activation = activation_function
        self.batch_norm = batch_norm
        if batch_norm:
            self.batchnorm = BatchNorm()
        self.p = dropout_p
        self.name = 'FC'
        
    def initialize_Wb(self):
        before_dim = self.X.shape[1]
        self.W = np.random.randn(before_dim, self.dim) / np.sqrt(before_dim) # see notes above
        self.b = np.random.randn(self.dim).reshape(1, self.dim) # see notes above
        
    def forward_propagation(self):
        if not self.isfirst:
            self.X = self.before.Z
        
        if not hasattr(self, 'W'):
            self.initialize_Wb() 
            
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
            
class Softmax_Classifier(FC):
    def __init__(self, activation_function, num_neurons, batch_norm = False, dropout_p = 1.0):
        super().__init__(activation_function, num_neurons, batch_norm, dropout_p)
        self.name = 'Softmax'
        
    def calculate_loss(self):
        loss = self.activation.forward_loss(self.Z, self.y)
        return loss
            
    def predict(self):
        return self.activation.predict(self.Z)
    
    def calculate_acc(self): 
        pred = self.predict()
        return sum( pred == self.y ) / len(self.y)    

class CNN(BaseLayer):
    def __init__(self, activation_function, params):
        super().__init__()
        self.activation = activation_function
        self.conv = Convolution()
        self.pooling = Pooling()
        self.name = 'CNN'
       
        self.num_filters,  self.filter_h, self.filter_w = params['num_filters'], params['filter_h'],params['filter_w']
        self.filter_stride, self.filter_padding = params['filter_stride'],params['filter_padding']
        self.pooling_size, self.pooling_stride = params['pooling_size'],params['pooling_stride']
    
    def initialize_Wb(self):
        # W: (num_filters, num_channels, filter_h, filter_w)
        # X: (num_examples, num_channels, height, width)
        # b: (num_filters, 1)
        if self.isfirst:
            self.num_channels = self.X.shape[-3] 
        else:
            self.num_channels = self.before.num_filters
        self.W = np.random.randn(self.num_filters, self.num_channels, self.filter_h, self.filter_w) / np.sqrt(self.num_filters / 2.)
        self.b = np.random.randn(self.num_filters).reshape(self.num_filters,1)
        
    def forward_propagation(self):
        if not self.isfirst:
            self.X = self.before.Z

        if not hasattr(self, 'W'):
            self.initialize_Wb() 
            
        self.D = self.conv.forward(self.X, self.W, self.b, 
                                   stride = self.filter_stride, 
                                   padding = self.filter_padding)
        self.Z_conv = self.activation.forward(self.D)
        self.Z_conv_pool = self.pooling.forward(self.Z_conv, 
                                                size = self.pooling_size, 
                                                stride = self.pooling_stride)
        
        if self.after.name != 'CNN':
            self.Z_pool_flat = self.Z_conv_pool.ravel().reshape(self.X.shape[0], -1)
            self.Z = self.Z_pool_flat
        else:
            self.Z = self.Z_conv_pool
    
    def backward_propagation(self):
        if self.after.name != 'CNN':
            self.dZ_conv_pool = self.dZ.ravel().reshape(self.Z_conv_pool.shape)
        else:
            self.dZ_conv_pool = self.dZ
            
        self.dZ_conv = self.pooling.backward(self.dZ_conv_pool)
        self.dX, self.dW, self.db  = self.conv.backward(self.dZ_conv)
        
        if not self.isfirst:
            self.before.dZ = self.dX