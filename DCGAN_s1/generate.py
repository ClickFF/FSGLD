import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from models import build_generator
from properties import calc_metrics
from data_loader import FingerprintData
import os

os.makedirs('outputs', exist_ok=True)

# load the saved model 
generator = build_generator(100)
generator.load_weights('./bestmodel/saved_best_model.h5')

# generate samples
noise = tf.random.normal([10000, 100])
generated = generator(noise).numpy().reshape(-1, 168)

data_path = 'data/combine' 
try:
    real_data = FingerprintData(data_path, 'pos').x.reshape(-1, 168)
except FileNotFoundError:
    print(f"Error: Cannot find data file at {data_path}.json")
    print("Please ensure:")
    print("1. The data file exists")
    print("2. The path is correct")
    exit(1)

# evaluation 
metrics = calc_metrics(generated, real_data)

# save the results
pd.DataFrame(np.round(generated)).to_csv('outputs/generated_samples.csv', index=False)
print("Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
