import numpy as np
from script.Layer import *
class Network:
    def __init__(self):
        self.layers = []
        self.input = []
        self.y = []
        
    def add(self, new_layer):
        if self.layers:
            self.layers[-1].after = new_layer
            new_layer.before = self.layers[-1]
        self.layers.append(new_layer)
    
    def load_data(self, input, y):
        self.layers[0].set_first_layer(input)
        self.layers[-1].set_last_layer(y)
        
    def initialize(self, input, y, batch_size):
        self.input = input
        self.y = y
        self.load_data(input[:batch_size,:], y[:batch_size])
        #for layer in self.layers:
        #    layer.initialize_Wb()

    def train(self, num_iter, learning_rate, batch_size, rand_, lambda_, optimizer = 'Vanilla', Val_X = None, Val_y = None, 
             CAL_STEP = 100, PRINT_STEP = 100):
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
        for i in range(num_iter):    
            # Calculate batch index
            if not rand_:
                idx = list(range(self.input.shape[0]))
            else:
                idx = np.random.randint(self.input.shape[0], size = batch_size)
            
            self.load_data(self.input[idx,:], self.y[idx])
            
            # Forward Propagation
            for layer in self.layers:
                layer.forward_propagation()
                
            # Print Traing Acc/Loss
            if (i % CAL_STEP == 0):
                t_loss = self.layers[-1].calculate_loss()
                t_acc  = self.layers[-1].calculate_acc()
                train_loss.append(t_loss)
                train_acc.append(t_acc)
                
            if (i % PRINT_STEP == 0):
                print('Train at Iter {0:2d}: loss - {1:.3f}, Acc - {2:.3f}'.format(i, t_loss, t_acc))
                
            # Backward Propagation
            for layer in self.layers[::-1]:
                layer.backward_propagation()
                layer.update_weight(learning_rate, lambda_ = lambda_ , method = optimizer)
            
            # Print Validation Acc/Loss
            if (i % CAL_STEP == 0 and Val_X is not None):
                v_acc, v_loss = self.evaluate(Val_X, Val_y)
                val_loss.append(v_loss)
                val_acc.append(v_acc) 
                
            if (i % PRINT_STEP == 0 and Val_X is not None):
                print('Validation at Iter {0:2d}: loss - {1:.3f}, Acc - {2:.3f}'.format(i, v_loss, v_acc))

        # Finally return loss list
        return train_loss, train_acc, val_loss, val_acc
    
    def predict(self, X):
        self.load_data(X, y = None)
        for layer in self.layers:
            layer.forward_propagation()
        return layers[-1].predict()
        
    def evaluate(self, X, y):
        self.load_data(X, y)
        for layer in self.layers:
            layer.forward_propagation()
        loss = self.layers[-1].calculate_loss()
        acc  = self.layers[-1].calculate_acc()
        return acc, loss