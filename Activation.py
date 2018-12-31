import numpy as np

class Tanh:
    def calc(self, x):
        return np.tanh(x)
    
    def derv(self,x):
        return 4/((np.exp(-x)+np.exp(x))**2)
    
    def name(self):
        return 'Tanh'
    
class Relu:
    def calc(self,x):
        if x > 0 :
            return x
        else: 
            return 0
    
    def derv(self,x):
        if x > 0 :
            return 1
        else: 
            return 0        
    
    def name(self):
        return 'RelU'
    
class Sigmoid:
    def calc(self,x):
        return (1/(1+np.exp(-x)))
    
    def derv(self,x):
        return np.exp(-x)/((np.exp(-x)+1)**2)
    
    def name(self):
        return 'Sigmoid'
    
def activeFunc(x):
    return {
        'tanh' : Tanh(),
        'Relu' : Relu(),
        'Sigmoid': Sigmoid()
    }[x]