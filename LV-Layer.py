import numpy as np
from Activation import activeFunc


## Version control:
## Version 1.0: Batch wont compile
version = 1.0
print("Layer version " + str(version))

class Layer:
    def __init__(self, insize, outsize , actvfunc , init_range, R_min=-1, R_max=1 ):
        ## init_range - the range for the initial w
        ## delta (batch_sizexoutput)
        ## v (1xinput)
        ## w (input x output)
        self.R_min = R_min
        self.R_max = R_max
        self.w = init_range * np.random.uniform(R_min,R_max,[insize , outsize]) ## To-Do initializie weights to R available
        self.insize = insize;
        self.outsize = outsize;
        self.actvfunc = actvfunc
        self.v = 0
        self.out = 0
        self.delta = 0
        self.first_update = True
    
    def calc(self,input1):
        try:
            #input_bias = np.append(input1, [1])
            self.v = input1 @ self.w
            self.out = self.actvfunc.calc(input1 @ self.w)
        except ValueError:
            #input_bias = np.append(input1, [1]).transpose()
            input1 = input1.transpose()
            self.v = input1 @ self.w
            self.out = self.actvfunc.calc(input1 @ self.w)
            

    def delta_calc(self, layer, batchsize,iter):
        if self.first_update == True:
            self.delta = np.zeros([1, self.outsize])
            self.delta[0,:] = self.actvfunc.derv(self.v)*(layer.delta @ layer.w.transpose())
            self.first_update = False
        else:
            self.delta = np.append(self.delta, self.actvfunc.derv(self.v)*((layer.delta @ layer.w.transpose())),0)
    
    def update_w(self,eta):
        self.w = self.w +eta*self.out.transpose() @ self.delta.transpose()
        self.delta = 0
        self.first_update = True
        