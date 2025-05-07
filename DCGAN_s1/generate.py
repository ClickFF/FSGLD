import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from models import build_generator
from properties import calc_metrics
from data_loader import FingerprintData
import os

# 确保输出目录存在
os.makedirs('outputs', exist_ok=True)

# 加载模型
generator = build_generator(100)
generator.load_weights('./bestmodel/saved_best_model.h5')

# 生成样本
noise = tf.random.normal([10000, 100])
generated = generator(noise).numpy().reshape(-1, 168)

# 加载真实数据 - 使用完整路径
data_path = 'data/combine' 
try:
    real_data = FingerprintData(data_path, 'pos').x.reshape(-1, 168)
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