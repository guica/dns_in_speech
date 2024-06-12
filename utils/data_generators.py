from abc import ABC, abstractmethod

import numpy as np
from pesq import pesq
import pystoi
import pandas as pd

from utils import somar_sinais, add_white_gaussian_noise, calculate_stft_magnitude_and_phase, reconstruct_signal_from_stft, calculate_snr


class DataGenerator(ABC):
    def __init__(self, sound_files, noise_files, block_size=8, normalize_phi=True, return_snr=False):
        self.sound_files = sound_files
        self.noise_files = noise_files
        self.normalize_phi = normalize_phi
        self.block_size = block_size
        self.return_snr = return_snr

    def pick_random_blocks(self, batch_size):
            
        if batch_size % self.block_size != 0:
            raise ValueError(f"O tamanho do lote deve ser um múltiplo de {self.block_size}")

        # Calcula quantos blocos de 8 existem nos dados fornecidos
        num_blocks = batch_size // self.block_size
        
        # Escolhe blocos aleatórios de sons e ruídos
        sound_block_indices = np.random.choice(self.sound_files.shape[0] // self.block_size, size=num_blocks, replace=False) * self.block_size
        noise_block_indices = np.random.choice(self.noise_files.shape[0] // self.block_size, size=num_blocks, replace=False) * self.block_size

        # Seleciona os arquivos de sons e ruídos
        sound_batch = np.array([self.sound_files[i:i+8] for i in sound_block_indices]).reshape(-1, self.sound_files.shape[-1])
        noise_batch = np.array([self.noise_files[i:i+8] for i in noise_block_indices]).reshape(-1, self.noise_files.shape[-1])
        
        # Verifica se reshape não excedeu a quantidade de amostras disponível, ajustando se necessário
        if len(sound_batch) > batch_size:
            sound_batch = sound_batch[:batch_size]
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]

        return sound_batch, noise_batch

    def normalize_and_add_noise(self, sound, noise, add_white=True):
        min_valor = np.min(sound)
        max_valor = np.max(sound)
        
        # Defina o novo intervalo desejado
        novo_min = -0.4
        novo_max = 0.4
        
        # Realize a escala do sinal para o novo intervalo
        sound_escalado = (sound - min_valor) / (max_valor - min_valor) * (novo_max - novo_min) + novo_min

        potencia_sound = np.mean(np.abs(sound_escalado) ** 2.0)
        potencia_noise = np.mean(np.abs(noise) ** 2.0)

        if potencia_sound > 0. and potencia_noise > 0.:
            sr = np.random.randint(0, 20, size=(1,)[0])
            noisy_sound = somar_sinais(sound_escalado, noise, sr)

        elif potencia_sound > 0.:
            noisy_sound = sound_escalado

        else:
            # raise ValueError(f"A potência do sinal de voz é {potencia_sound}. A adição de ruído não pode ser computada")
            return None, None

        if add_white:
            noisy_sound = add_white_gaussian_noise(noisy_sound, np.random.randint(15, 30, size=(1,)[0]))
        
        noisy_sound = np.clip(noisy_sound, -1.0, 1.0)

        return sound_escalado, noisy_sound
    
    def assemble_phasors(self, A, phi):
        if self.normalize_phi:
            # Monta o fasor e normaliza a faze entre 0-1
            F = np.concatenate(
                [ A.reshape(A.shape[0], A.shape[1], 1), (phi.reshape(phi.shape[0], phi.shape[1], 1) / (2*np.pi)) + 0.5 ],
                axis=-1
            )
        else:
            # Monta o fasor
            F = np.concatenate(
                [ A.reshape(A.shape[0], A.shape[1], 1), phi.reshape(phi.shape[0], phi.shape[1], 1) ],
                axis=-1
            )
        
        return F
    
    def disassemble_phasors(self, F):
        A = F[:, :, 0]

        if self.normalize_phi:
            phi = (F[:, :, 1] - 0.5) * 2 * np.pi
        else:
            phi = F[:, :, 1]
        
        return A, phi
    
    def calcular_snr_db(self, sinal, sinal_com_ruido):
        """
        Calcula a Relação Sinal-Ruído (SNR) em decibéis.

        :param sinal: Array do sinal original.
        :param sinal_com_ruido: Array do sinal com ruído.
        :return: SNR em decibéis.
        """
        # Calcula o sinal de ruído
        ruido = sinal_com_ruido - sinal

        # Calcula a potência do sinal e do ruído
        potencia_sinal = np.mean(np.square(sinal))
        potencia_ruido = np.mean(np.square(ruido))

        # Evita a divisão por zero
        if potencia_ruido == 0:
            return 100.

        # Calcula a SNR em decibéis
        snr_db = 10 * np.log10(potencia_sinal / potencia_ruido)
        return snr_db

    @abstractmethod
    def generate_sample_completo(self, batch_size=32, only_return_mudule=False):
        pass


class NoisyTargetGenerator(DataGenerator):
    def __init__(self, sound_files, noise_files, block_size=8, normalize_phi=True, return_snr=False, add_white=True):
        super().__init__(sound_files, noise_files, block_size=block_size, normalize_phi=normalize_phi, return_snr=return_snr)
        self.add_white = add_white

    def generate_sample_completo(self, batch_size=32, include_clean=False, only_return_mudule=False):
        while True:
            # Carrega um lote de vozes e ruidos
            sound_batch, noise_batch = self.pick_random_blocks(batch_size)

            x_train = []
            y_train = []
            
            # Adiciona ruído a cada som e calcula a nota PESQ
            for sound, noise in zip(sound_batch, noise_batch):

                sound_escalado, noisy_sound = self.normalize_and_add_noise(sound, noise, add_white=self.add_white)
                
                if sound_escalado is None or noisy_sound is None:
                    continue
                
                try:
                    A, phi, _, _ = calculate_stft_magnitude_and_phase(sound_escalado)
                    A_noisy, phi_noisy, _, _ = calculate_stft_magnitude_and_phase(noisy_sound)
                except:
                    continue

                F = self.assemble_phasors(A, phi)
                F_noisy = self.assemble_phasors(A_noisy, phi_noisy)
                
                if not only_return_mudule:
                    # Adiciona os exemplos de fasores completos aos lotes de treinamento
                    x_train.append(F_noisy)
                    y_train.append(F)
                    
                    if include_clean:
                        x_train.append(F)
                        y_train.append(F)
                
                else:
                    # Adiciona os exemplos somente do modulo aos lotes de treinamento
                    if self.return_snr:
                        snr_matrix = np.ones(F_noisy.shape[0], F_noisy.shape[1], 1)
                        snr = self.calcular_snr_db(sound_escalado, noisy_sound)
                        snr_matrix = snr * snr_matrix

                        x_train.append(np.concatenate([F_noisy[..., 0].reshape(F_noisy.shape[0], F_noisy.shape[1], 1), snr_matrix], axis=-1))
                    else:
                        x_train.append(F_noisy[..., 0].reshape(F_noisy.shape[0], F_noisy.shape[1], 1))

                    y_train.append(F[..., 0].reshape(F.shape[0], F.shape[1], 1))
                    
                    if include_clean:
                        x_train.append(F[..., 0].reshape(F.shape[0], F.shape[1], 1))
                        y_train.append(F[..., 0].reshape(F.shape[0], F.shape[1], 1))

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            
            yield x_train, y_train

class PESQGenerator(DataGenerator):
    def __init__(self, sound_files, noise_files, model=None, block_size=8, normalize_phi=True, normalize_pesq=True):
        super().__init__(sound_files, noise_files, block_size=block_size, normalize_phi=normalize_phi)
        self.model = model
        self.normalize_pesq = normalize_pesq

    def norm_pesq(self, pesq_score):
        valor_min = 1.04
        valor_max = 4.64
        return (pesq_score - valor_min) / (valor_max - valor_min)

    def generate_sample_completo(self, batch_size=32):
        while True:
            # Carrega um lote de vozes e ruidos
            sound_batch, noise_batch = self.pick_random_blocks(batch_size)

            x_train = []
            y_train = []
            
            # Adiciona ruído a cada som e calcula a nota PESQ
            for sound, noise in zip(sound_batch, noise_batch):
                sound_escalado, noisy_sound = self.normalize_and_add_noise(sound, noise)

                if sound_escalado is None or noisy_sound is None:
                    continue

                #Calcula a nota PESQ
                try:
                    pesq_score = pesq(8000, sound_escalado, noisy_sound, 'nb')
                except:
                    continue

                if self.normalize_pesq:
                    pesq_score = self.norm_pesq(pesq_score)

                try:
                    A, phi, _, _ = calculate_stft_magnitude_and_phase(sound_escalado)
                    A_noisy, phi_noisy, _, _ = calculate_stft_magnitude_and_phase(noisy_sound)
                except:
                    continue

                F = self.assemble_phasors(A, phi)
                F_noisy = self.assemble_phasors(A_noisy, phi_noisy)

                # Adiciona o exemplo ao lote de treinamento
                x_train.append(F_noisy)
                x_train.append(F)

                y_train.append(pesq_score)

                if self.normalize_pesq:
                    y_train.append(1.0)
                else:
                    y_train.append(4.64)

                if self.model:
                    x_generated = self.model.predict(F_noisy.reshape(1, *F_noisy.shape), verbose=False)
                    try:
                        A, phi = self.disassemble_phasors(x_generated[0])
                        gen_signal = reconstruct_signal_from_stft(A, phi)
                        
                        pesq_gen = pesq(8000, sound, gen_signal, 'nb')

                        if self.normalize_pesq:
                            pesq_gen = self.normalize_pesq(pesq_gen)
                        
                        x_train.append(x_generated[0])
                        y_train.append(pesq_gen)

                    except:
                        pass

            x_train = np.array(x_train)
            y_train = np.array(y_train).reshape(-1, 1)
            
            yield x_train, y_train


class NoisyTargetDoubleGenerator(NoisyTargetGenerator):
    def __init__(self, sound_files, noise_files, block_size=8, normalize_phi=True):
        super().__init__(sound_files, noise_files, block_size=block_size, normalize_phi=normalize_phi)

    def generate_sample_completo(self, batch_size=32):
        while True:
            # Carrega um lote de vozes e ruidos
            sound_batch, noise_batch = self.pick_random_blocks(batch_size)

            x_train = []
            y_train = []
            
            # Adiciona ruído a cada som e calcula a nota PESQ
            for sound, noise in zip(sound_batch, noise_batch):

                sound_escalado, noisy_sound = self.normalize_and_add_noise(sound, noise)
                
                if sound_escalado is None or noisy_sound is None:
                    continue
                
                try:
                    A, phi, _, _ = calculate_stft_magnitude_and_phase(sound_escalado)
                    A_noisy, phi_noisy, _, _ = calculate_stft_magnitude_and_phase(noisy_sound)
                except:
                    continue

                F = self.assemble_phasors(A, phi)
                F_noisy = self.assemble_phasors(A_noisy, phi_noisy)
                
                # Adiciona os exemplos aos lotes de treinamento
                x_train.append(F_noisy)
                y_train.append(F)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            x_module = x_train[..., 0]
            x_phase = x_train[..., 1]
            x_module = x_module[..., np.newaxis]
            x_phase = x_phase[..., np.newaxis]

            y_module = y_train[..., 0]
            y_phase = y_train[..., 1]
            y_module = y_module[..., np.newaxis]
            y_phase = y_phase[..., np.newaxis]
            
            yield [x_module, x_phase], [y_module, y_phase]


class NoisyTargetWithMetricsGenerator(DataGenerator):
    def __init__(self, sound_files, noise_files, block_size=8, normalize_phi=True):
        super().__init__(sound_files, noise_files, block_size=block_size, normalize_phi=normalize_phi)

    def generate_sample_completo(self, batch_size=32, only_return_mudule=False):
        while True:
            sound_batch, noise_batch = self.pick_random_blocks(batch_size)
            
            x_train = []
            y_train = []

            x_wave = []
            y_wave = []

            metrics_data = []
            
            # Adiciona ruído a cada som e calcula a nota PESQ
            for sound, noise in zip(sound_batch, noise_batch):
                # Normaliza a voz e adiciona ruído
                sound_escalado, noisy_sound = self.normalize_and_add_noise(sound, noise, add_white=True)
                
                if sound_escalado is None or noisy_sound is None:
                    continue
                
                try:
                    A, phi, _, _ = calculate_stft_magnitude_and_phase(sound_escalado)
                    A_noisy, phi_noisy, _, _ = calculate_stft_magnitude_and_phase(noisy_sound)
                except:
                    continue

                # Calcula a nota PESQ
                try:
                    pesq_score = pesq(8000, sound_escalado, noisy_sound.reshape(-1), 'nb')
                except:
                    continue
                
                # Calcula o score STOI
                stoi_score = pystoi.stoi(sound_escalado, noisy_sound, 8000)
                
                # Calcula SNR
                snr = calculate_snr(sound_escalado, noisy_sound)

                # Calcula o ID
                # ID = itakura_distortion(sound_escalado, noisy_sound, 256, 11)
                
                metrics_data.append([pesq_score, stoi_score, snr])
                
                F = self.assemble_phasors(A, phi)
                F_noisy = self.assemble_phasors(A_noisy, phi_noisy)
                
                # Adiciona os exemplos aos lotes de treinamento

                if only_return_mudule:
                    x_train.append(F_noisy[..., 0].reshape(F_noisy.shape[0], F_noisy.shape[1], 1))
                    y_train.append(F[..., 0].reshape(F.shape[0], F.shape[1], 1))
                
                else:
                    x_train.append(F_noisy)
                    y_train.append(F)

                x_wave.append(noisy_sound)
                y_wave.append(sound_escalado)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            x_wave = np.array(x_wave)
            y_wave = np.array(y_wave)

            metrics_df = pd.DataFrame(metrics_data, columns=['PESQ', 'STOI', 'SNR'])

            yield (x_wave, y_wave), (x_train, y_train) , metrics_df


class PESQWithMetricsGenerator(DataGenerator):
    def __init__(self, sound_files, noise_files, model=None, block_size=8, normalize_phi=True, normalize_pesq=True):
        super().__init__(sound_files, noise_files, block_size=block_size, normalize_phi=normalize_phi)
        self.model = model
        self.normalize_pesq = normalize_pesq

    def norm_pesq(self, pesq_score):
        valor_min = 1.04
        valor_max = 4.64
        return (pesq_score - valor_min) / (valor_max - valor_min)

    def generate_sample_completo(self, batch_size=32, only_return_mudule=False):
        while True:
            # Carrega um lote de vozes e ruidos
            sound_batch, noise_batch = self.pick_random_blocks(batch_size)

            x_train = []
            y_train = []
            metrics_data = []
            
            # Adiciona ruído a cada som e calcula a nota PESQ
            for sound, noise in zip(sound_batch, noise_batch):
                sound_escalado, noisy_sound = self.normalize_and_add_noise(sound, noise)

                if sound_escalado is None or noisy_sound is None:
                    continue

                #Calcula a nota PESQ
                try:
                    pesq_score = pesq(8000, sound_escalado, noisy_sound, 'nb')
                except:
                    continue
                
                # Calcula SNR
                snr = calculate_snr(sound_escalado, noisy_sound)

                if self.normalize_pesq:
                    pesq_score = self.norm_pesq(pesq_score)

                try:
                    A, phi, _, _ = calculate_stft_magnitude_and_phase(sound_escalado)
                    A_noisy, phi_noisy, _, _ = calculate_stft_magnitude_and_phase(noisy_sound)
                except:
                    continue

                F = self.assemble_phasors(A, phi)
                F_noisy = self.assemble_phasors(A_noisy, phi_noisy)

                # Adiciona o exemplo ao lote de treinamento
                x_train.append(F_noisy)
                x_train.append(F)

                y_train.append(pesq_score)

                if self.normalize_pesq:
                    y_train.append(1.0)
                else:
                    y_train.append(4.64)

                metrics_data.append([pesq_score, snr, False])
                metrics_data.append([4.64, np.inf, False])

                if self.model:
                    x_generated = self.model.predict(F_noisy.reshape(1, *F_noisy.shape), verbose=False)
                    try:
                        A, phi = self.disassemble_phasors(x_generated[0])
                        gen_signal = reconstruct_signal_from_stft(A, phi)
                        
                        pesq_gen = pesq(8000, sound_escalado, gen_signal, 'nb')

                        if self.normalize_pesq:
                            pesq_gen = self.norm_pesq(pesq_gen)
                        
                        x_train.append(x_generated[0])
                        y_train.append(pesq_gen)

                        snr_gen = calculate_snr(sound_escalado[:gen_signal.shape[0]], gen_signal)
                        metrics_data.append([pesq_gen, snr_gen, True])

                    except:
                        print('error with model')
                        pass

            x_train = np.array(x_train)
            y_train = np.array(y_train).reshape(-1, 1)
            metrics_df = pd.DataFrame(metrics_data, columns=['PESQ - Real', 'SNR', 'Generated'])
            
            yield x_train, y_train, metrics_df