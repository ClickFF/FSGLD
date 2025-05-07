import numpy as np
import pandas as pd
import tensorflow as tf

class CB2DataLoader:
    def __init__(self, active_path, inactive_path):
        self.active_path = active_path
        self.inactive_path = inactive_path
        self.load_data()
        
    def load_data(self):
        cb2_active = pd.read_csv(self.active_path, header=None, sep='\t')
        active_data = np.array([[[int(j)] for j in cb2_active.loc[i,1]] for i in cb2_active.index])  # 保持 int
        active_labels = np.ones(len(active_data)) 

    # 读取非活性分子（同上）
        cb2_inactive = pd.read_csv(self.inactive_path, header=None, sep='\t')
        inactive_data = np.array([[[int(j)] for j in cb2_inactive.loc[i,1]] for i in cb2_inactive.index])
        inactive_labels = np.zeros(len(inactive_data))

    # 合并数据
        self.x = np.concatenate((active_data, inactive_data), axis=0)
        self.y = np.concatenate((active_labels, inactive_labels)).reshape(-1, 1)
        self._shuffle()
    
    def _shuffle(self):
        idx = np.random.permutation(len(self.x))
        self.x = self.x[idx]
        self.y = self.y[idx]
        
    def get_dataset(self, batch_size=32):
        return tf.data.Dataset.from_tensor_slices((self.x, self.y)) \
                             .batch(batch_size) \
                             .prefetch(tf.data.AUTOTUNE)