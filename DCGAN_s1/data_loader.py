import numpy as np
import json
import tensorflow as tf

class FingerprintData:
    def __init__(self, file_path, label_type):
        self.file_path = file_path  
        self.label_type = label_type  # 'pos' or 'neg'
        if self.label_type == 'pos':
            self.load_pos_data()
        elif self.label_type == 'neg':
            self.load_neg_data()
        self.preprocess()
        
    def load_pos_data(self):
        with open('{}.json'.format(self.file_path), 'r') as file:  
            temp = json.load(file)
        for i in range(len(temp['y'])):
            if temp['y'][i] == [1,0]:
                temp['y'][i] = [1]
            else:
                temp['y'][i] = [0]
        positive_index = [i for i in range(len(temp['y'])) if temp['y'][i] == [1]]
        
        self.y = np.array([temp['y'][i] for i in positive_index])
        self.x = np.array([temp['x'][i] for i in positive_index])
                
        self.x = self.x.reshape(self.x.shape[0], self.x.shape[1],1)
        self.z = self.x.reshape(self.x.shape[0], self.x.shape[1])
        self.m = len(positive_index)
        self.n = self.x.shape[1]

    def load_neg_data(self):
        with open('{}.json'.format(self.file_path), 'r') as file:  
            temp = json.load(file)
        negative_index = [i for i in range(len(temp['y'])) if temp['y'][i] == [0]]
        self.y = np.array([temp['y'][i] for i in negative_index])
        self.x = np.array([temp['x'][i] for i in negative_index])
        self.x = self.x.reshape(self.x.shape[0], self.x.shape[1],1)
        self.z = self.x.reshape(self.x.shape[0], self.x.shape[1])
        self.m = len(negative_index)
        self.n = self.x.shape[1]

    def preprocess(self):
        self._shuffle()
        
    def _shuffle(self):
        idx = np.random.permutation(len(self.x))
        self.x = self.x[idx]
        self.y = self.y[idx]
        
    def get_dataset(self, batch_size=32):
        return tf.data.Dataset.from_tensor_slices((self.x, self.y)) \
                             .batch(batch_size) \
                             .prefetch(tf.data.AUTOTUNE)