import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from models import build_generator, to_categorical
from properties import calc_metrics
from data_loader import CB2DataLoader
import os

# 确保输出目录存在
os.makedirs('outputs', exist_ok=True)
mcs_df = pd.read_csv('data/cb2maccsii.csv', header=None, sep='\s+')
mcs_df = mcs_df.set_index(mcs_df.columns[0])
mcs_index = [i for i in mcs_df.columns if mcs_df.loc['structureii', i] == 1]

umask = np.ones((168, 1), dtype=np.float32)
mask = np.zeros((168, 1), dtype=np.float32)
for i in (np.array(mcs_index) - 1):
    umask[i, 0] = 0
    mask[i, 0] = 1

# === 加载模型并传入 mask ===
generator = build_generator(100, 2, umask=umask, mask=mask)
generator.load_weights('./bestmodel/saved_best_model.h5')

# 生成样本
num_samples = 10000
latent_dim = 100
noise = tf.random.normal([num_samples, latent_dim])
target_labels = tf.constant([1] * num_samples)
target_one_hot = to_categorical(target_labels)  # shape: (10000, 2)
noise_with_condition = tf.concat([noise, target_one_hot], axis=1)  # shape: (10000, 102)
sample = generator(noise_with_condition, training=False)  # shape: (10000, 168, 1)
generated = tf.reshape(sample, [num_samples, -1]).numpy()  # (10000, 168)

active_path = 'data/cb2_active.csv'
inactive_path = 'data/cb2_inactive.csv'
try:
    data_loader = CB2DataLoader(active_path, inactive_path)
    real_data = data_loader.x[data_loader.y[:, 0] == 1].reshape(-1, 168)
except FileNotFoundError:
    print(f"Error: Cannot find data file at {data_path}.json")
    print("Please ensure:")
    print("1. The data file exists")
    print("2. The path is correct")
    exit(1)

# 计算指标
metrics = calc_metrics(generated, real_data)

# 保存结果
pd.DataFrame(np.round(generated)).to_csv('outputs/generated_samples.csv', index=False)
print("Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")