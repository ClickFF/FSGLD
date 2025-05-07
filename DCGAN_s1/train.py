import tensorflow as tf
from models import build_generator, build_discriminator
from data_loader import FingerprintData
import matplotlib.pyplot as plt
import os

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.data = FingerprintData(config['data_path'], config['label_type'])
        self.dataset = self.data.get_dataset(config['batch_size'])
        
        self.generator = build_generator(config['latent_dim'])
        self.discriminator = build_discriminator(config['fingerprint_dim'])
        
        self.g_optimizer = tf.keras.optimizers.Adam(
            config['g_lr'], beta_1=0.5, beta_2=0.999)
        self.d_optimizer = tf.keras.optimizers.Adam(
            config['d_lr'], beta_1=0.5, beta_2=0.999)
        
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.config['latent_dim']])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator(noise, training=True)
            
            # Discriminator outputs
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            
            # Loss calculations
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = (real_loss + fake_loss) / 2
            
            # Generator loss
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        
        # Apply gradients
        gradients = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables))
        
        gradients = gen_tape.gradient(
            g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss
    
    def train(self):
        for epoch in range(self.config['epochs']):
            for batch, (real_data, _) in enumerate(self.dataset):
                d_loss, g_loss = self.train_step(real_data)
            
            if epoch % self.config['sample_interval'] == 0:
                self._save_checkpoint(epoch)
                self._generate_samples(epoch)
                
            print(f"Epoch {epoch}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    def _save_checkpoint(self, epoch):
        self.generator.save_weights(
            f"{self.checkpoint_dir}/generator_{epoch}.h5")
    
    def _generate_samples(self, epoch):
        noise = tf.random.normal([1, self.config['latent_dim']])

        sample = self.generator(noise, training=False)
        
        plt.plot(sample.numpy().reshape(-1), label='Generated')
        plt.plot(self.data.x[0,:,0], label='Real')
        plt.title(f"Epoch {epoch}")
        plt.legend()
        plt.savefig(f"{self.checkpoint_dir}/sample_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    config = {
        'data_path': 'data/combine',
        'label_type': 'pos',
        'latent_dim': 100,
        'fingerprint_dim': 168,
        'batch_size': 32,
        'epochs': 1000,
        'g_lr': 0.0001,
        'd_lr': 0.000001,
        'sample_interval': 50,
        'checkpoint_dir': 'models'
    }
    
    trainer = GANTrainer(config)
    trainer.train()
