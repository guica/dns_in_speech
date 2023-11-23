import scipy.io.wavfile as wavfile
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool
import random
import librosa

class Sound(object):

    sounds_path = None
    noise_sounds_path = None
    base_shape_size = None

    noise_sounds = None
    clean_sounds = None

    TOO_SHORT_ERROR = 'Shape too short'

    def __init__(self, sounds_path, noise_sounds_path, base_shape_size, limit=25000, pattern='.wav', seed=13):
        # Define a seed
        np.random.seed(seed)
        random.seed(seed)

        self.sounds_path = sounds_path
        self.noise_sounds_path = noise_sounds_path
        self.base_shape_size = base_shape_size

        noise_files = self.get_sounds_from_folder(noise_sounds_path, pattern)
        clean_files = self.get_sounds_from_folder(sounds_path, pattern)

        if len(clean_files) == 0 or len(noise_files) == 0:
            raise ValueError('Directory does not contain sounds with pattern {}'.format(pattern))

        if len(clean_files) > limit:
            random.shuffle(clean_files)
            clean_files = clean_files[:limit]
        
        if len(noise_files) > limit:
            random.shuffle(noise_files)
            noise_files = noise_files[:limit]

        # Criar filas para armazenar os sons
        noise_queue = Queue()
        clean_queue = Queue()

        # Criar processos para carregar os sons
        noise_process = Process(target=self.load_sounds_in_queue, args=(noise_files, noise_queue, 'Loading Noise Files'))
        clean_process = Process(target=self.load_sounds_in_queue, args=(clean_files, clean_queue, 'Loading Speech Files'))

        # Iniciar os processos
        noise_process.start()
        clean_process.start()

        clean_sounds = []
        while True:
            sound = clean_queue.get()
            if sound is None:
                break
            clean_sounds.append(sound)

        clean_sounds = [sound for sound in clean_sounds if sound != self.TOO_SHORT_ERROR]
        random.shuffle(clean_sounds)
        self.clean_sounds = np.concatenate(clean_sounds, axis=0)

        # Obter os sons das filas
        noise_sounds = []
        while True:
            sound = noise_queue.get()
            if sound is None:
                break
            noise_sounds.append(sound)

        noise_sounds = [sound for sound in noise_sounds if sound != self.TOO_SHORT_ERROR]
        random.shuffle(noise_sounds)
        self.noise_sounds = np.concatenate(noise_sounds, axis=0)

        # Aguardar a finalização dos processos
        noise_process.join()
        clean_process.join()

        # Dividir os sons em conjuntos de treinamento, validação e teste
        # train_split = 0.8
        # val_split = 0.1

        # self.train_X = self.clean_sounds[:int(len(self.clean_sounds) * train_split)]
        # self.val_X = self.clean_sounds[int(len(self.clean_sounds) * train_split):int(len(self.clean_sounds) * (train_split + val_split))]
        # self.test_X = self.clean_sounds[int(len(self.clean_sounds) * (train_split + val_split)):]

    def load_sounds_in_queue(self, sound_files, queue, message):
        with Pool() as pool:
            results = list(tqdm(pool.imap(self.load_sound_file, [sound_file for sound_file in sound_files]), total=len(sound_files), desc=message))

        for result in results:
            queue.put(result)

        # Sinalizar o fim da fila
        queue.put(None)

    def load_sound_file(self, sound_file):
        sound = self.load_wav(sound_file)
        step = 4000
        sound = sound[step:]
        if sound.shape[0] < self.base_shape_size:
            # print(sound.shape)
            # print(sound_file)
            return 'Shape too short'
        else:
            sound = librosa.util.frame(sound, frame_length=self.base_shape_size, hop_length=int(self.base_shape_size / 2.), axis=0)
            # sound = sound[:len(sound) // self.base_shape_size * self.base_shape_size].reshape(-1, self.base_shape_size)
            return sound

    def get_sounds_from_folder(self, path, pattern):
        max_depth = 4  # replace with the maximum depth of subfolders to search

        sound_list = []

        for root, dirs, files in os.walk(path):
            depth = root[len(path) + len(os.path.sep):].count(os.path.sep)
            if depth < max_depth:
                for file in files:
                    if file.endswith(pattern):#or file.endswith('.bin'):
                        wav_path = os.path.join(root, file)
                        sound_list.append(wav_path)
        
        return sound_list

    def load_wav(self, filename, debug=False):
        # Load the WAV file
        sample_rate, data = wavfile.read(filename)
        
        if debug==True:
            print(np.max(data))
            print(np.min(data))

        # Normalize the data to float32 values between -1 and 1
        normalized_sound = np.float32(data)/ float(32767.0)
        
        if debug==True:
            print(np.max(normalized_sound))
            print(np.min(normalized_sound))

        return normalized_sound
