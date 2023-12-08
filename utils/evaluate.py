from abc import ABC, abstractmethod
import os

from pesq import pesq
import pystoi
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from keras import backend as K

from utils import calculate_snr, reconstruct_signal_from_stft
from data_generators import NoisyTargetWithMetricsGenerator, PESQWithMetricsGenerator
from sound import Sound

class Evaluator(ABC):
    sound_base = None
    model = None
    model_name = None
    data_generator = None
    SAVE_DIR = None

    def __init__(self, base_shape_size, speech_path, noise_path, model_path, model_name, save_dir='./metrics/'):
        self.sound_base = Sound(speech_path, noise_path, base_shape_size)
        self.model_name = model_name
        self.model = load_model(model_path, custom_objects={"K": K})
        self.SAVE_DIR = save_dir

    def stft_to_signal(self, stft, sampling_rate=8000, window_size=255, overlap=128):
        A = stft[..., 0]
        phi = stft[..., 1]
        signal = reconstruct_signal_from_stft(A, phi, sampling_rate=sampling_rate, window_size=window_size, overlap=overlap)

        return signal

    @abstractmethod
    def process_batch(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

class NoisyTargetEvaluator(Evaluator):
    sound_base = None
    model = None
    model_name = None
    data_generator = None

    def __init__(self, base_shape_size, speech_path, noise_path, model_path, model_name):
        super().__init__(base_shape_size, speech_path, noise_path, model_path, model_name)

        self.data_generator = NoisyTargetWithMetricsGenerator(self.sound_base.clean_sounds, self.sound_base.noise_sounds)
    
    def process_batch(self, x_batch, y_batch, module_only=False):
        if module_only:
            stfts = self.model.predict(x_batch[..., 0], verbose=False)
            stfts = np.concatenate([stfts, x_batch[..., 1].reshape(-1, x_batch.shape[1], x_batch.shape[2], 1)], axis=-1)
        else:
            stfts = self.model.predict(x_batch, verbose=False)

        M = stfts.shape[0]  # Obtenha as dimensões do array de resultados do modelo

        pesq_scores = []
        stoi_scores = []
        snr_scores = []

        for i in range(M):
            filtered = stfts[i, :, :, :]  # Obtenha o resultado do modelo para a iteração atual

            clean = y_batch[i, :, :, :]  # Obtenha o sinal limpo correspondente

            clean_signal = self.stft_to_signal(clean).reshape(-1)
            filtered_signal = self.stft_to_signal(filtered).reshape(-1)

            try:
                pesq_score = pesq(8000, clean_signal, filtered_signal, 'nb')
            except:
                pesq_score = 1.04
            stoi_score = pystoi.stoi(clean_signal, filtered_signal, 8000)
            snr_score = calculate_snr(clean_signal, filtered_signal)

            pesq_scores.append(pesq_score)
            stoi_scores.append(stoi_score)
            snr_scores.append(snr_score)

        return pesq_scores, stoi_scores, snr_scores
    
    def evaluate(self, batch_num=50, module_only=False):
        results = []
        df_resultado = pd.DataFrame()

        for _ in tqdm(range(batch_num)):
            x_batch, y_batch, metrics_batch_df = next(self.data_generator.generate_sample_completo(batch_size=128))
            results.append((self.process_batch(x_batch, y_batch, module_only=module_only), metrics_batch_df))

        df_resultado = pd.DataFrame()

        for result , metrics_batch_df in results:
            pesq_scores, stoi_scores, snr_scores = result
            metrics_batch_df['PESQ (Filtered)'] = pesq_scores
            metrics_batch_df['STOI (Filtered)'] = stoi_scores
            metrics_batch_df['SNR (Filtered)'] = snr_scores
            
            df_resultado = pd.concat([df_resultado, metrics_batch_df], ignore_index=True)
        
        # Saving the results
        # Get the current datetime
        current_datetime = datetime.now()

        # Format the datetime as a string to use in the file name
        datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        
        if not os.path.exists(self.SAVE_DIR):
            # Se não existir, criar o diretório
            os.makedirs(self.SAVE_DIR)

        # Define the file name with the datetime
        file_name = f"{self.SAVE_DIR + self.model_name}-metrics_{datetime_str}.xlsx"
        df_resultado.to_excel(file_name, index=False)
        print('File saved to {}'.format(file_name))

        return df_resultado
    

class PESQEvaluator(Evaluator):
    sound_base = None
    model = None
    model_name = None
    data_generator = None
    gen_model = None

    def __init__(self, base_shape_size, speech_path, noise_path, model_path, model_name, gen_model=None):
        super().__init__(base_shape_size, speech_path, noise_path, model_path, model_name)

        self.gen_model = gen_model
        self.data_generator = PESQWithMetricsGenerator(self.sound_base.clean_sounds, self.sound_base.noise_sounds, model=self.gen_model, normalize_pesq=False)
    
    def process_batch(self, x_batch):
        predicted_pesq_scores = self.model.predict(x_batch, verbose=False)
        print(predicted_pesq_scores.shape)
        pesq_scores = predicted_pesq_scores.flatten().tolist()
        print(len(pesq_scores))
        return pesq_scores
    
    def evaluate(self, batch_num=50):
        results = []
        df_resultado = pd.DataFrame()

        for _ in tqdm(range(batch_num)):
            x_batch, _, metrics_batch_df = next(self.data_generator.generate_sample_completo(batch_size=128))
            results.append((self.process_batch(x_batch), metrics_batch_df))

        for pesq_scores , metrics_batch_df in results:
            metrics_batch_df['PESQ - Predicted'] = pesq_scores
            df_resultado = pd.concat([df_resultado, metrics_batch_df], ignore_index=True)
        
        # Saving the results
        # Get the current datetime
        current_datetime = datetime.now()

        # Format the datetime as a string to use in the file name
        datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Define the file name with the datetime
        file_name = f"{self.model_name}-metrics_{datetime_str}.xlsx"
        df_resultado.to_excel(file_name, index=False)
        print('File saved to {}'.format(file_name))

        return df_resultado