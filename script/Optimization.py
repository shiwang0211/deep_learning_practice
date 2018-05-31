import numpy as np
# Treat all elements of dX as a whole
#  Intuition: 
#  If gradient direction not changed, increase update, faster convergence
#  If gradient direction changed, reduce update, reduce oscillation

def VanillaUpdate(x, dx, learning_rate):
    x += -learning_rate * dx
    return x

def MomentumUpdate(x, dx, v, learning_rate, mu):
    v = mu * v - learning_rate * dx # integrate velocity, mu's typical value is about 0.9
    x += v # integrate position     
    return x, v
# Treat each element of dX adaptively
# Intuition:
# 1. Those dx receiving infrequent updates should have higher learning rate. vice versa 
# 2. We don't want: the gradients accumulate, and the learning rate monotically decrease, 
# 2. We want: modulates the learning rate of each weight based on the magnitudes of its gradient
# 3. Still want to use "momentum-like" update to get a smooth gradient

# 1. AdaGrad
def AdaGrad(x, dx, learning_rate, cache, eps):
    cache += dx**2
    x += - learning_rate * dx / (np.sqrt(cache) + eps) # (usually set somewhere in range from 1e-4 to 1e-8)
    return x, cache
    
# 1+2. RMSprop
def RMSprop(x, dx, learning_rate, cache, eps, decay_rate): #Here, decay_rate typical values are [0.9, 0.99, 0.999]
    cache = decay_rate * cache + (1 - decay_rate) * dx**2
    x += - learning_rate * dx / (np.sqrt(cache) + eps)
    return x, cache
    
# 1+2+3. Adam
def Adam(x, dx, learning_rate, m, v, t, beta1, beta2, eps):
    m = beta1*m + (1-beta1)*dx # Smooth gradient
    #mt = m / (1-beta1**t) # bias-correction step
    v = beta2*v + (1-beta2)*(dx**2) # keep track of past updates
    #vt = v / (1-beta2**t) # bias-correction step
    x += - learning_rate * m / (np.sqrt(v) + eps) # eps = 1e-8, beta1 = 0.9, beta2 = 0.999   
    return x, m, v
class WeightUpdate:
    def __init__(self, init_value):
        self.val = init_value
        self.cache = np.zeros_like(self.val, dtype=np.float64)
        self.m = np.zeros_like(self.val, dtype=np.float64)
        self.v = np.zeros_like(self.val, dtype=np.float64)
        self.t = 0
    
    def Update(self, d, learning_rate, lambda_ , method):
        
        old_val = self.val
        if method == 'Vanilla':
            self.val = VanillaUpdate(self.val, d, learning_rate)
        elif method == 'MomentumUpdate':
            self.val, self.v = MomentumUpdate(self.val, d, self.v, learning_rate, mu = 0.9)
        elif method == 'AdaGrad':
            self.val, self.cache = AdaGrad(self.val, d, learning_rate, self.cache, eps = 1e-5)
        elif method == 'RMSprop':
            self.val, self.cache = AdaGrad(self.val, d, learning_rate, self.cache, eps = 1e-5, decay_rate = 0.99)
        elif method == 'Adam':
            self.val, self.m, self.v = Adam(self.val, d, learning_rate, self.m, self.v, self.t, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)  
            self.t += 1
            
        # Regularization
        self.val -= lambda_ * old_val
        return self.val