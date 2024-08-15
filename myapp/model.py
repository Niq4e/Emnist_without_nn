import os
import numpy as np
import pickle

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model_lgbm.pkl')
        labels_path = os.path.join('myapp', 'labelz.pickle')

        with open(labels_path, 'rb') as f:
            self.labelz = pickle.load(f)

        with open(model_path, 'rb') as m:
            self.model = pickle.load(m)

    def predict(self, x): 
        data = x.copy()
        data = data.reshape(1,-1)
        data = data.astype(np.float32)
        data /= data.max()
        
        predict = self.model.predict(data)[0]
        symbol = self.labelz[predict]
        
        return symbol

        
        
        
