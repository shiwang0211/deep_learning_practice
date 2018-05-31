import numpy as np
from script.Optimization import *
class Mul:
    def forward(self, W, X):
        return np.dot(X, W)
    
    def backward(self, W, X, dWX):
        dW = np.dot( np.transpose(X), dWX )
        dX = np.dot( dWX, np.transpose (W))
        return dW, dX
class Add:
    def forward(self, WX, b):
        return WX + b

    def backward(self, WX, b, dS):
        dWX = dS * np.ones_like(WX, dtype=np.float64)
        db = np.dot(np.ones((1, dS.shape[0]), dtype=np.float64), dS)
        return db, dWX
class ReLU:
    def forward(self, S):
        Z = S * (S > 0)
        return Z
    
    def backward(self, S, dZ):
        dS = 1. * (S > 0) * dZ
        return dS
class Tanh:
    def forward(self, S):
        Z = np.tanh(S)
        return Z
    
    def backward(self, S, dZ):
        Z = self.forward(S)
        dS = (1.0 - np.square(Z)) * dZ
        return dS
class Sigmoid:
    def forward(self, S):
        Z = 1. / (1.0 + np.exp(-S))
        return Z
    
    def backward(self, S, dZ):
        Z = self.forward(S)
        dS =(1 - Z) * Z * dZ
        return dS
class  Softmax:
    # For Training
    def __init__(self):
        self.num_examples = 0
    
    def forward(self, S):
        self.num_examples = S.shape[0]
        exp_S = np.exp(S)
        Z = exp_S / np.sum(exp_S, axis = 1, keepdims = True)
        return Z

    def backward(self, S, y): # Note: y instead of dZ
        probs = Z = self.forward(S)
        for i in range(len(y)):
            true_label = y[i]
            probs[i][true_label] -= 1 # see equation above
        dS = probs
        return dS
    
    # For evaluation    
    def forward_loss(self, Z, y):
        probs = Z
        log_probs = []
        for prob, true_label in zip(probs, y):
            log_probs.append(np.log(prob[true_label]))
        avg_cross_entropy_loss = - 1. / self.num_examples * np.sum(log_probs) # see equation above
        return avg_cross_entropy_loss
    
    # For prediction
    def predict(self, Z):
        return np.argmax(Z, axis = 1)
class BatchNorm:
    def __init__(self):
        self.cache = ()
        
    def forward(self, X, gamma, beta, eps):
        num_examples = X.shape[0]
        
        mu_B = 1. / num_examples * np.sum(X, axis = 0)
        X_mu = X - mu_B
        var_B = 1. / num_examples * np.sum(  X_mu ** 2, axis = 0 )
        sqrt_var_B = np.sqrt(var_B + eps)
        i_sqrt_var_B = 1. / sqrt_var_B
        X_hat =  X_mu * i_sqrt_var_B
        gammaX = gamma * X_hat
        DZ = gammaX + beta
        
        self.cache = (X_hat, X_mu, gamma, i_sqrt_var_B, sqrt_var_B, var_B, eps)
        return DZ
    
    def backward(self, dDZ):
        num_examples = dDZ.shape[0]
        X_hat, X_mu, gamma, i_sqrt_var_B, sqrt_var_B, var_B, eps = self.cache
        
        # scale and shift
        dbeta = np.sum(dDZ, axis = 0)
        dgammaX = dDZ
        dgamma = np.sum(dgammaX * X_hat, axis = 0)
        dXhat = dgammaX * gamma
        
        # Standardize
        di_sqrt_var_B = np.sum(dXhat * X_mu, axis = 0)
        d_x_mu_2 = dXhat * i_sqrt_var_B
        dsqrt_var_B = -1. / (sqrt_var_B ** 2) * di_sqrt_var_B
        dvar_B = 0.5 * 1. / np.sqrt(var_B + eps) * dsqrt_var_B

        # Batch variance
        dsquare = 1. / num_examples * np.ones_like(dDZ) * dvar_B
        d_x_mu_1 = 2 * X_mu * dsquare
        
        # Batch mean
        d_x_mu = d_x_mu_2 + d_x_mu_1 # d(f(x)g(x)) = f(x)g'(x) = f'(x)g(x)
        dmu = -1. * np.sum(d_x_mu, axis = 0)
        dx_2 = d_x_mu
        dx_1 = 1. / num_examples * np.ones_like(dDZ) * dmu
        dx = dx_2 + dx_1
        
        return dx, dgamma, dbeta
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