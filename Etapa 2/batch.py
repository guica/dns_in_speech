import sys
sys.path.insert(0, '/tf/utils/')

from pesq import pesq
import numpy as np
import pandas as pd
import pystoi

from utils import calculate_snr, somar_sinais, add_white_gaussian_noise, calculate_snr, itakura_distortion


# def itakura_distortion(x, y):
#     log_x = 10 * np.log10(np.abs(x) ** 2)
#     log_y = 10 * np.log10(np.abs(y) ** 2)
#     distortion = np.sum((log_x - log_y) ** 2)
#     return distortion

class DataGenerator:
    def __init__(self, sound_files, noise_files):
        self.sound_files = sound_files
        self.noise_files = noise_files
        self.MIN_NOISE_LEVEL = 0.
        self.MAX_NOISE_LEVEL = 20
        self.MIN_WHITE_GAUSS_LEVEL = 15.
        self.MAX_WHITE_GAUSS_LEVEL = 25.

    def generate_sample_metrics(self, window, order, batch_size=32):
        while True:
            # Carrega um lote de sons
            sound_batch_choices = np.random.choice(self.sound_files.shape[0], size=batch_size, replace=False)
            sound_batch = self.sound_files[sound_batch_choices]
            
            # Carrega um lote de ruídos
            noise_batch_choices = np.random.choice(self.noise_files.shape[0], size=batch_size, replace=False)
            noise_batch = self.noise_files[noise_batch_choices]
            
            x_train = []
            y_train = []
            metrics_data = []  # Lista para armazenar métricas
            
            for sound, noise in zip(sound_batch, noise_batch):
                
                sr = float(np.random.randint(self.MIN_NOISE_LEVEL, self.MAX_NOISE_LEVEL))
                noisy_sound = somar_sinais(sound, noise, sr)

                sr_gauss = float(np.random.randint(self.MIN_WHITE_GAUSS_LEVEL, self.MAX_WHITE_GAUSS_LEVEL))
                noisy_sound = add_white_gaussian_noise(noisy_sound, sr_gauss)
                noisy_sound = np.clip(noisy_sound, -1.0, 1.0)
                
                # Calcula a nota PESQ
                try:
                    pesq_score = pesq(8000, sound, noisy_sound.reshape(-1), 'nb')
                except:
                    continue
                
                # Calcula o score STOI
                stoi_score = pystoi.stoi(sound, noisy_sound, 8000)
                
                # Calcula SNR
                snr = calculate_snr(sound, noisy_sound)

                # Calcula o ID
                ID = itakura_distortion(sound, noisy_sound, window, order)
                
                x_train.append(noisy_sound)
                y_train.append(sound)
                
                metrics_data.append([sr, sr_gauss, pesq_score, stoi_score, snr, ID])
            
            # Cria um DataFrame com as métricas
            metrics_df = pd.DataFrame(metrics_data, columns=['SNR Ruído aditivo', 'SNR Ruído Gauss Branco', 'PESQ', 'STOI', 'SNR', 'ID'])
            
            yield np.array(x_train), np.array(y_train), metrics_df