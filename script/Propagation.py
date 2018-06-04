
import numpy as np
from util.im2col import *
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
class Softmax:
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
# Reference: https://github.com/wiseodd/hipsternet/blob/master/hipsternet
class Convolution:
    def __init__(self):
        self.cache = ()
        
    def forward(self, X, W, b, stride=1, padding=1):
        
        # W: (num_filters, num_channels, filter_h, filter_w)
        # X: (num_examples, num_channels, height, width)
        n_filters, d_filter, h_filter, w_filter = W.shape
        n_x, d_x, h_x, w_x = X.shape
        
        # Calculate Output Shape
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1
        
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)
        
        # Column-ize W and X
        # W_col: ( num_filters, num_channels * filter_h * filter_w) 
        # X_col: ( num_channels * filter_h * filter_w, h_out * w_out * num_examples)
        
        W_col = W.reshape(n_filters, -1)
        X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
        
        # Matrix Multiply
        # W_col * X_col: (num_filters, h_out * w_out * num_examples)
        # b: (num_filters, 1)
        # WX + b: (num_filters, h_out * w_out * num_examples)
        
        out = np.matmul(W_col, X_col) + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        # out: (num_examples, num_filters, h_out, w_out)
        self.cache = (X, W, b, stride, padding, X_col)

        return out


    def backward(self, dout):
        X, W, b, stride, padding, X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = np.matmul(dout_reshaped,X_col.T)
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = np.matmul(W_reshape.T,dout_reshaped)
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

        return dX, dW, db
class Maxpool:
    def forward(self, X_col):
        # X_reshaped: (num_examples * num_filters, 1, new_h_out, new_w_out)
        # for example: X.shape = (10,5,4,4), size = stride = 2, (new) h_out = w_out = 2
        # for example: X_reshaped = (50, 1, 2, 2)
        # X_col: (1 * size * size, num_filters(new_num_channels) * h_out * w_out * num_examples)
        # for example, X_col = (1*2*2, 5*2*2*10) = (4, 200)
        # max_idx.shape = out.shape = (200, )
        max_idx = np.argmax(X_col, axis=0) # for every (size * size) cells, select one max [0,1,2,3,1,2,...]
        out = X_col[max_idx, range(X_col.shape[1])]
        return out, max_idx

    def backward(self, dX_col, dout_col, max_idx):
        # Only max value got local gradient = 1
        dX_col[max_idx, range(dout_col.size)] = 1.0 * dout_col
        return dX_col
class Pooling:
    def __init__(self, pool_fun = Maxpool()):
        self.cache = ()
        self.max_idx = ()
        self.pool_fun = pool_fun
        
    def forward(self, X, size = 2, stride = 2):
        # Calculate new shape after pooling
        
        # X_shape = (num_examples, num_filters, h_out, w_out) from CNN layer
        # for example: X = (10,5,4,4), size=stride=2, h_out=w_out=2
        n, d, h, w = X.shape
        h_out = (h - size) / stride + 1
        w_out = (w - size) / stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        # X_reshaped: (num_examples * num_filters, new_h_out, new_w_out)
        X_reshaped = X.reshape(n * d, 1, h, w)
        
        # X_col: (num_filters(new_num_channels) * size * size, h_out, w_out * num_examples)
        X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

        out, self.max_idx = self.pool_fun.forward(X_col)
 
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)
        # out: (num_examples, num_filters (new_num_channels), new_h_out, new_w_out)
        
        self.cache = (X, size, stride, X_col)

        return out  
    
    
    def backward(self, dout):
        X, size, stride, X_col = self.cache
        n, d, w, h = X.shape

        dX_col = np.zeros_like(X_col)
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        dX = self.pool_fun.backward(dX_col, dout_col, self.max_idx)
        dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
        dX = dX.reshape(X.shape)

        return dX