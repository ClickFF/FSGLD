import tensorflow as tf
from models import build_generator, build_discriminator, concatenate_data_label, to_categorical
from data_loader import CB2DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class ConditionalGANTrainer:
    def __init__(self, config):
        self.config = config
        MCS = pd.read_csv(config['mask_path'], header=None, sep='\s+')
        MCS = MCS.set_index(MCS.columns[0])
        MCS_index = []
        for i in MCS.columns:
            if MCS.loc['structure3', i] == 1:
                MCS_index.append(i)

        global umask, mask
        umask = np.ones((168,1))
        for i in (np.array(MCS_index)-1):
            umask[i,0] = 0
    
        mask = np.zeros((168,1))
        for i in (np.array(MCS_index)-1):
            mask[i,0] = 1

        self.data = CB2DataLoader(config['active_path'], config['inactive_path'])
        self.dataset = self.data.get_dataset(config['batch_size'])
        
        self.generator = build_generator(config['latent_dim'], umask=umask, mask=mask)
        self.discriminator = build_discriminator(config['fingerprint_dim'])
        
        self.g_optimizer = tf.keras.optimizers.Adam(config['g_lr'], beta_1=0.5, beta_2=0.999)
        self.d_optimizer = tf.keras.optimizers.Adam(config['d_lr'], beta_1=0.5, beta_2=0.999)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @tf.function
    def train_step(self, real_data, labels):
        batch_size = tf.shape(real_data)[0]
        noise_z = tf.random.normal([batch_size, 1, self.config['latent_dim']])  # (N,1,100)
        labels_one_hot = to_categorical(labels)  # (N,2)
        labels_one_hot = tf.expand_dims(labels_one_hot, axis=1)  # (N,1,2)

        noise_with_condition = tf.concat([noise_z, labels_one_hot], axis=2)  # (N,1,102)
        noise_with_condition = tf.reshape(noise_with_condition, [batch_size, -1])  # (N,102)
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_data = self.generator(noise_with_condition, training=True)
        
            real_conditional = concatenate_data_label(real_data, labels) 
            fake_conditional = concatenate_data_label(fake_data, labels)
        
            real_output = self.discriminator(real_conditional, training=True)
            fake_output = self.discriminator(fake_conditional, training=True)
        
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss
        
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
    
        gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
    
        gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
    
        return d_loss, g_loss
    
    def train(self):
        for epoch in range(self.config['epochs']):
            for batch, (real_data, labels) in enumerate(self.dataset):
                d_loss, g_loss = self.train_step(real_data, labels)
            
            if epoch % self.config['sample_interval'] == 0:
                self._save_checkpoint(epoch)
                self._generate_samples(epoch)
                
            print(f"Epoch {epoch}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    def _save_checkpoint(self, epoch):
        self.generator.save_weights(f"{self.checkpoint_dir}/generator_{epoch}.h5")
    
    def _generate_samples(self, epoch, target_label=1):
        noise = tf.random.normal([1, self.config['latent_dim']], dtype=tf.float32)  # 显式指定 float32
        target_one_hot = to_categorical(target_label)  # (1, 2)
        noise_with_condition = tf.concat([noise, target_one_hot], axis=1)  # [1,102]
        sample = self.generator(noise_with_condition, training=False)
        
        plt.figure(figsize=(10, 5))
        plt.plot(sample.numpy().reshape(-1), label='Generated')
        plt.plot(self.data.x[0,:,0], label='Real')
        plt.title(f"Epoch {epoch} - Conditional Samples")
        plt.legend()
        plt.savefig(f"{self.checkpoint_dir}/sample_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    config = {
        'active_path': 'data/cb2_active.csv',
        'inactive_path': 'data/cb2_inactive.csv',
        'mask_path': 'data/cb2maccsii.csv',
        'latent_dim': 100,
        'fingerprint_dim': 168,
        'batch_size': 32,
        'epochs': 1000,
        'g_lr': 0.0001,
        'd_lr': 0.000001,
        'sample_interval': 50,
        'checkpoint_dir': 'models'
    }
    
    trainer = ConditionalGANTrainer(config)
    trainer.train()