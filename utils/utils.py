from scipy.signal import butter, filtfilt, resample
import scipy.signal
import scipy.linalg
import scipy.io.wavfile as wavfile
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa

def get_sounds_from_folder(path, pattern):
    max_depth = 3  # replace with the maximum depth of subfolders to search

    sound_list = []

    for root, dirs, files in os.walk(path):
        depth = root[len(path) + len(os.path.sep):].count(os.path.sep)
        if depth < max_depth:
            for file in files:
                if file.endswith(pattern):#or file.endswith('.bin'):
                    wav_path = os.path.join(root, file)
                    sound_list.append(wav_path)
    
    return sound_list

def load_wav(filename, debug=False):
    # Load the WAV file
    sample_rate, data = wavfile.read(filename)
    
    if debug==True:
        print(np.max(data))
        print(np.min(data))

    # Normalize the data to float32 values between -1 and 1
    normalized_sound = np.float32(data)/ 32767.0
    
    if debug==True:
        print(np.max(normalized_sound))
        print(np.min(normalized_sound))

    return normalized_sound

    # Convert the data to float32 values between -1 and 1
    data = np.float32(data / 32767.0)
    return data

def save_sound_to_wav(array, bin_path):
    # Set the sampling rate and audio data
    sampling_rate = 8000

    s = (32768.0*array.copy()).astype(np.int16)

    # Save the audio data as a WAV file
    wavfile.write(bin_path, sampling_rate, s)

def add_white_gaussian_noise(signal, snr):
    # Calculate the signal power and convert to dB
    signal_power = np.mean(signal**2.0)
    signal_power_db = 10.0 * np.log10(signal_power)

    # Calculate the noise power required for the specified SNR and convert to linear scale
    noise_power_db = signal_power_db - snr
    noise_power = 10.0**(noise_power_db / 10.0)

    # Generate random noise with the required power and add it to the signal
    noise = np.random.normal(0.0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise

    return np.float32(noisy_signal)

def generate_white_gaussian_noise(signal):
    # Generate random noise
    noise = np.random.normal(0, 1, len(signal))

    return noise


def somar_sinais(sinal1, sinal2, relacao_potencia_dB):
    # Verificar se os sinais têm o mesmo tamanho
    if len(sinal1) != len(sinal2):
        raise ValueError("Os sinais devem ter o mesmo tamanho")

    # Calcular a potência do primeiro sinal
    potencia_sinal1 = np.mean(np.abs(sinal1) ** 2.0)

    # Converter a relação de potência para escala linear
    relacao_potencia_linear = 10.0 ** (relacao_potencia_dB / 10.0)

    # Calcular a potência desejada para o ruído
    potencia_desejada_ruido = potencia_sinal1 / relacao_potencia_linear

    # Calcular a potência do segundo sinal (ruído)
    potencia_sinal2 = np.mean(np.abs(sinal2) ** 2.0)

    # Ajustar a amplitude do segundo sinal (ruído)
    fator_amplitude = np.sqrt(potencia_desejada_ruido / potencia_sinal2)
    sinal2_ajustado = sinal2 * fator_amplitude

    # Somar os sinais
    sinal_somado = sinal1 + sinal2_ajustado

    return np.float32(sinal_somado)

def calcular_componentes_fourier(sinal):
    componentes = np.fft.fft(sinal)
    # Obter a magnitude máxima dos componentes
    # max_magnitude = np.max(np.abs(componentes), axis=1).reshape(-1, 1)
    # Normalizar os componentes para o intervalo [-1, 1]
    max_magnitude = 1
    # componentes_normalizados = componentes / max_magnitude
    
    res = np.stack((np.real(componentes), np.imag(componentes)), axis=2)

    return np.array(res), max_magnitude

def reconstruir_sinal(componentes, max_magnitude):
    comp = componentes[:, :, 0] + 1j * componentes[:, :, 1]
    print(comp.shape)
    sinal_reconstruido = np.fft.ifft(comp) * max_magnitude
    return np.real(sinal_reconstruido)

def undersample_signal_with_antialiasing(sound, orig_sr, desired_sr):
    
    new_sound = sound.copy()
    
    nyq = 0.5 * orig_sr
    cutoff_freq = desired_sr /2
    order = 6
    normal_cutoff = cutoff_freq / nyq

    # Compute the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    sound = filtfilt(b, a, new_sound)

    target_size = int(len(new_sound)*desired_sr/orig_sr)
    downsampled_signal = resample(new_sound, target_size)

    return downsampled_signal

def load_sound_from_bin(bin_path):
    # Carrega arquivo inteiro formato DOS
    # de nome contido na string str na variável x

    # with open(bin_path, 'rb') as f:
    #     x = np.fromfile(f, dtype=np.uint16)

    # supu = 2 ** 16  # 'supremo' unsigned
    # supc = 2 ** 15  # 'supremo' complemento de dois

    # nx = len(x)
    # x = 256 * x[1:nx:2] + x[0:nx:2]
    # i = np.where(x >= supc)[0]  # amostras que devem ser complementadas
    # x[i] = x[i] - supu

    # # return x, i
    # return uint16_to_int16(x)

    with open(bin_path, 'rb') as f:
        # Read the binary data as a string
        data = f.read()

    # Convert the string to a numpy int16 array
    array = np.frombuffer(data, dtype=np.int16)
    return array.astype(np.float32) / 32768.0

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate the signal-to-noise ratio (SNR) in dB between a clean signal and a noisy signal.
    
    Args:
    clean_signal (numpy array): clean signal of shape (N, 1)
    noisy_signal (numpy array): noisy signal of shape (N, 1)
    
    Returns:
    snr_db (float): signal-to-noise ratio (SNR) in dB
    """
    clean_signal = clean_signal.reshape(-1, 1)
    noisy_signal = noisy_signal.reshape(-1, 1)
    
    # Calculate the power of the clean signal
    clean_power = np.mean(clean_signal ** 2)
    
    # Calculate the power of the noise signal
    noise_signal = noisy_signal - clean_signal
    noise_power = np.mean(noise_signal ** 2)
    
    # Calculate the SNR in dB
    snr_db = 10 * np.log10(clean_power / noise_power)
    
    return snr_db

def calculate_snrseg(clean_signal, noisy_signal, segment_size):
    """
    Calculate the signal-to-noise ratio (SNR) using the SNRseg metric between a clean signal and a noisy signal.
    
    Args:
    clean_signal (numpy array): clean signal of shape (N, 1)
    noisy_signal (numpy array): noisy signal of shape (N, 1)
    segment_size (int): size of each segment
    
    Returns:
    snr_db (float): signal-to-noise ratio (SNR) using the SNRseg metric
    """
    clean_signal = clean_signal.reshape(-1,1)
    noisy_signal = noisy_signal.reshape(-1,1)
    
    # Calculate the number of segments
    num_segments = int(np.floor(len(clean_signal) / segment_size))
    
    # Calculate the power of the clean signal
    clean_power = np.mean(clean_signal ** 2)
    
    # Initialize the noise power and segment count
    noise_power = 0
    segment_count = 0
    
    # Calculate the power of the noise signal for each segment
    for i in range(num_segments):
        segment_start = i * segment_size
        segment_end = (i+1) * segment_size
        noise_signal = noisy_signal[segment_start:segment_end] - clean_signal[segment_start:segment_end]
        noise_power += np.mean(noise_signal ** 2)
        segment_count += 1
        
    # Calculate the average noise power over all segments
    noise_power = noise_power / segment_count
    
    # Calculate the SNR using the SNRseg metric
    snr_db = 10 * np.log10(clean_power / noise_power)
    
    return snr_db

# def lpc_analysis(signal, order):
#     # Aplicar janelamento de Hamming
#     windowed_signal = signal * np.hamming(len(signal))

#     # Calcular os coeficientes LPC usando autocoeficientes
#     autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
#     r = autocorr[len(autocorr) // 2:len(autocorr) // 2 + order + 1]
    
#     # Usar a função toeplitz para realizar a decomposição de Levinson-Durbin
#     A = scipy.linalg.solve_toeplitz((r[:-1], r[1:]), -r[1:])
    
#     # Calcular o erro de predição linear (residual)
#     residual = signal - scipy.signal.lfilter(np.hstack([1, -A]), 1, signal)
    
#     # Calcular a variância do erro de predição linear (residual)
#     var_residual = np.var(residual)

#     return A, var_residual, r

def preemphasis_filter(signal, preemphasis_coeff):
    # Aplicar filtro de pré-ênfase de alta passagem
    preemphasized_signal = np.append(signal[0], signal[1:] - preemphasis_coeff * signal[:-1])
    return preemphasized_signal

def lpc_analysis(signal, order):
    # Aplicar janelamento de Hamming
    # windowed_signal = signal * np.hamming(len(signal))
    # Aplicar pré-ênfase no sinal de entrada
    windowed_signal = preemphasis_filter(signal, 0.83)
    windowed_signal = windowed_signal * np.hanning(len(windowed_signal))

    # Calcular os coeficientes LPC usando autocoeficientes
    autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
    r = autocorr[len(autocorr) // 2:len(autocorr) // 2 + order + 1]
    
    # Usar a função toeplitz para realizar a decomposição de Levinson-Durbin
    A = scipy.linalg.solve_toeplitz((r[:-1], r[1:]), -r[1:])

    # Crie a nova matriz A na forma [1, a1, a2, a3, ...]
    G = A[0]
    A /= G
    # Calcular o erro de predição linear (residual)
    residual = signal - scipy.signal.lfilter(np.hstack([1.0/G, -A]),1.0, signal)
    
    # Calcular a variância do erro de predição linear (residual)
    var_residual = np.var(residual)

    return A, np.sqrt(var_residual), r

def autocorrelation_matrix(signal, order):
    """
    Calcula a matriz de autocorrelação de um sinal.
    
    Args:
    signal (numpy.array): O sinal de entrada.
    order (int): A ordem da matriz de autocorrelação.

    Returns:
    numpy.array: A matriz de autocorrelação de tamanho (order + 1) x (order + 1).
    """
    N = len(signal)
    autocorr_matrix = np.zeros((order + 1, order + 1))
    
    for i in range(order + 1):
        for j in range(order + 1):
            if i <= j:
                autocorr_matrix[i, j] = np.sum(signal[:N - j] * signal[j - i:N - i])
            else:
                autocorr_matrix[i, j] = autocorr_matrix[j, i]
    
    return autocorr_matrix

def itakura_distortion(sinal_original, sinal_ruidoso, window, order):
    """
    Calcula a distorção de Itakura-Saito entre um sinal original e um sinal ruidoso divididos em janelas.

    Args:
    sinal_original (numpy.array): O sinal de áudio original.
    sinal_ruidoso (numpy.array): O sinal de áudio ruidoso para comparação.
    window (int): O tamanho da janela para dividir os sinais.
    order (int): A ordem do modelo LPC para análise.

    Returns:
    float: O valor médio do Índice de Distorção (ID) para todas as janelas analisadas.
           Quanto maior o valor, maior o índice de distorção presente.
    """
    ids = []

    n_windows = sinal_original.shape[0] // window
    s = sinal_original[:n_windows*window]
    r = sinal_ruidoso[:n_windows*window]
    
    for i in range(n_windows):
        frame_s = s[i*window:(i + 1)*window]
        frame_r = r[i*window:(i + 1)*window]

        ad, _, _ = lpc_analysis(frame_r, order)
        ac, _, x = lpc_analysis(frame_s, order)

        Rc = autocorrelation_matrix(frame_s, order)

        ad = np.pad(ad, (1, 0), mode='constant', constant_values=1).reshape(1, order + 1)
        ac = np.pad(ac, (1, 0), mode='constant', constant_values=1).reshape(1, order + 1)

        numerador = ad @ Rc @ ad.T
        denominador = ac @ Rc @ ac.T
        
        id = np.log10(numerador/denominador)
        ids.append(id)
    
    return np.mean(ids)

def performance(df_results, filter_name, snr_lte=100., snr_gte=0.):
    df_results_filtered = df_results[df_results['SNR'] <= snr_lte]
    df_results_filtered = df_results_filtered[df_results_filtered['SNR'] >= snr_gte]
    metrics = ['PESQ', 'STOI', 'SNR', 'ID']
    
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        
        x = df_results_filtered[metric]
        y = df_results_filtered[f'{metric} (Filtered)']
    
        if metric != 'ID':
            improved = x < y  # Verifica se o filtro melhorou a métrica
        else:
            improved = x > y  # Verifica se o filtro melhorou a métrica
        
        plt.scatter(x[improved], y[improved], c='green', alpha=0.5, label='Melhorado')
        plt.scatter(x[~improved], y[~improved], c='red', alpha=0.5, label='Piorado')
        
        # Adiciona a linha f(x) = x
        plt.plot(x, x, 'b--', label='Limiar')
        
        plt.xlabel(f'{metric} - Ruidoso')
        plt.ylabel(f'{metric} - Filtrado')
        plt.title(f'{filter_name} - {metric}')
        plt.legend()
        
        plt.show()

def preemphasis(signal, alpha=0.97):
    """
    Aplica o filtro de pré-ênfase a um sinal de áudio.
    
    Args:
        signal (numpy array): O sinal de áudio de entrada.
        alpha (float): Coeficiente de pré-ênfase (padrão: 0.97).

    Retorna:
        numpy array: O sinal de saída após a pré-ênfase.
    """
    # Aplica o filtro de pré-ênfase
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    
    return emphasized_signal

def calculate_f0(signal, sample_rate, min_f0=70., max_f0=300.):
    """
    Calcula a F0 usando o algoritmo AMDF (Average Magnitude Difference Function).

    Args:
        signal (numpy array): O sinal de áudio de entrada.
        sample_rate (int): A taxa de amostragem do sinal.
        min_f0 (int): Frequência fundamental mínima esperada em Hz (padrão: 85 Hz).
        max_f0 (int): Frequência fundamental máxima esperada em Hz (padrão: 255 Hz).

    Retorna:
        float: A F0 estimada em Hz.
    """
    # Defina os limites da busca de F0
    min_period = int(sample_rate / max_f0)
    max_period = int(sample_rate / min_f0)

    amdf = np.zeros(max_period - min_period + 1)

    # Calcule o AMDF para diferentes atrasos (períodos)
    for delay in range(min_period, max_period + 1):
        diff = np.abs(signal[:-delay] - signal[delay:])
        amdf[delay - min_period] = np.sum(diff)

    # Encontre o atraso que minimiza o AMDF (representando a F0)
    estimated_period = np.argmin(amdf) + min_period

    # Calcula a F0 em Hz
    f0 = sample_rate / estimated_period

    return f0

def vad(signal, sample_rate, min_f0=70., max_f0=300.):
    # Calcula a F0 do sinal
    f0 = calculate_f0(signal, sample_rate, min_f0=min_f0, max_f0=max_f0)
    
    # Verifica se a F0 está dentro da faixa desejada
    is_voice = min_f0 <= f0 <= max_f0

    # Aplica um limiar para determinar a atividade de voz
    return is_voice if is_voice else False

def calculate_stft_magnitude_and_phase(signal, sampling_rate=8000, window_size=255, overlap=128):
    # Calcula a STFT usando a biblioteca librosa
    stft_result = librosa.stft(signal, n_fft=window_size, hop_length=overlap)
    
    magnitude, phase = librosa.magphase(stft_result)
    phi = np.angle(phase)
    f = librosa.fft_frequencies(sr=sampling_rate, n_fft=window_size)
    t = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sampling_rate, hop_length=overlap)

    return magnitude, phi, f, t

def reconstruct_signal_from_stft(magnitude, phi, sampling_rate=8000, window_size=255, overlap=128):
    # Reconstruct the signal from magnitude and phase
    complex_spec = magnitude * np.exp(1j * phi)
    signal = librosa.istft(complex_spec, hop_length=overlap)

    return signal