import numpy as np
import sys

sys.path.append('/home/frcvision1/Final/learning-to-drive-in-a-day-carla-0.9/gan_vae')
from vae_gan import VaeGan


class GanVaeController:
    def __init__(self, batch_size=64, latent_dim=512, log_dir='./vaeganlogs/logs/celeba_test', learnrate_init=1e-5,
                 image_size=(80, 160, 3), kl_tolerance=0.5, epoch_per_optimization=10, buffer_size=1000):
        # GAN_VAE Shapes
        self.z_size = latent_dim
        self.image_size = image_size

        # Learning params
        self.learning_rate = learnrate_init
        self.kl_tolerance = kl_tolerance

        # Train_params
        self.epochs_per_optimization = epoch_per_optimization
        self.batch_size = batch_size

        # VAE Data Buffer
        self.buffer_size = buffer_size
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_reset()

        self.gan_vae = VaeGan(batch_size=batch_size, log_dir=log_dir, learning_rate=learnrate_init)
        self.target_gan_vae = VaeGan(batch_size=1, is_training=False)

    def buffer_reset(self):
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer = np.zeros((self.buffer_size,
                                self.image_size[0],
                                self.image_size[1],
                                self.image_size[2]),
                               dtype=np.uint8)

    def buffer_append(self, arr):
        assert arr.shape == self.image_size
        self.buffer_pos += 1
        if self.buffer_pos > self.buffer_size - 1:
            self.buffer_pos = 0
            self.buffer_full = True
        self.buffer[self.buffer_pos] = arr

    def buffer_get_copy(self):
        if self.buffer_full:
            return self.buffer.copy()
        return self.buffer[:self.buffer_pos]

    def encode(self, arr):
        assert arr.shape == self.image_size
        arr.astype(np.float) / 255.0
        arr = arr.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])
        return self.target_gan_vae.encode(arr)

    def decode(self, arr):
        assert arr.shape == (1, self.z_size)
        arr = self.target_gan_vae.decode(arr)
        arr *= 255.0
        return arr

    def optimize(self):
        ds = self.buffer_get_copy()
        num_batches = int(np.floor(len(ds) / self.batch_size))
        for epoch in range(self.epochs_per_optimization):
            np.random.shuffle(ds)
            for idx in range(num_batches):
                batch = np.array(ds[idx * self.batch_size:(idx + 1) * self.batch_size])
                obs = batch.astype(np.float) / 255.0
                feed = {self.gan_vae.input_batch: obs}
                self.gan_vae.train(feed)
        self.set_target_params()

    def set_target_params(self):
        params, param_names = self.gan_vae.get_encoder_params()
        self.target_gan_vae.set_encoder_params(params, param_names)

    def save(self, path):
        self.target_gan_vae.save_encoder_json(path)

    def load(self, path):
        self.target_gan_vae.load_encoder_json(path)


if __name__ == '__main__':
    GanVaeController()
