import time
import numpy as np
from Activation import activeFunc
from Layer import Layer
from Tests import sets
from Tests import Set
from prettytable import PrettyTable
import progressbar
        
class NeuralNetwork():
    def __init__(self,init_range,eta):
        self.init_range = init_range
        self.layers = []
        self.eta = eta
        self.errors = 0
        
    def add_layer(self,insize,outsize,actvfunc='tanh'):
        self.layers.append(Layer(insize, outsize , activeFunc(actvfunc) , self.init_range))
    
    def forward_pass(self,inputdata):
        input_x = inputdata
        for x in self.layers:
            x.calc(input_x)
            input_x = x.out
    def back_propogation(self,y,batchsize,iter):
        L = len(self.layers)
        self.layers[L-1].delta = (y - self.layers[L-1].out)*self.layers[L-1].actvfunc.derv(self.layers[L-1].v)
        for l in reversed(range(L-1)):
            self.layers[l].delta_calc(self.layers[l+1],batchsize,iter)
    
    def epoch(self, epochs, batchsize, x_train, y_train):
        errvec = np.zeros([1,3])
        self.errors = np.zeros([epochs*batchsize,1])
        iter = 0
        for j in progressbar.progressbar(range(epochs)):
            for i in range(batchsize):
                self.forward_pass(inputdata = x_train.current())
                errvec = self.layers[-1].out - y_train.current()
                self.errors[iter] = 0.5 * errvec @ errvec.transpose() 
                x_train.next_entry()
                self.back_propogation(y_train.current(),batchsize,i)
                y_train.next_entry()
                time.sleep(0.00001)
            
            for x in self.layers: x.update_w(self.eta)
        print("Done")
    def summary(self):
        ## summary table of the NN
        i=0
        t = PrettyTable(['Layer number', 'Output shape', 'ActiveFunc', 'Weights shape'])
        for x in self.layers:
            t.add_row([i, x.outsize, x.actvfunc, x.w.shape])
            i = i+1
        print(t)