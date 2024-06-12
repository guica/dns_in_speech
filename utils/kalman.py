import numpy as np
import librosa
import scipy

# Parâmetros do sistema
order = 11  # Ordem da análise LPC
window_size = 600  # Tamanho da janela
sample_rate = 8000  # Taxa de amostragem (exemplo)

def lpc_analysis(signal, order):
    # windowed_signal = preemphasis_filter(signal, 0.83)
    windowed_signal = signal * np.hanning(len(signal))

    a = librosa.lpc(windowed_signal, order=order)

    b = np.hstack([[0], -1 * a[1:]])
    y_hat = scipy.signal.lfilter(b, [1], signal)

    var_residual = np.var(signal - y_hat)

    return a, np.sqrt(var_residual), 0

def build_matrices(A, window_size):
    Ak = np.zeros((order, order))
    Ak[:, 0] = -A[1:]
    Ak[:-1, 1:] = np.eye(order - 1)

    H = np.zeros((1, order))
    H[0, 0] = 1.0

    return Ak, H

def kalman_filter(signal, Ak, H, Q, R):
    x_hat = np.zeros(order)  # Estado estimado
    P = np.eye(order)  # Covariância estimada

    filtered_signal = []

    for sample in signal:
        # Atualização temporal (Predição)
        x_hat = np.dot(Ak, x_hat)
        P = np.dot(np.dot(Ak, P), Ak.T) + Q

        # Atualização de mensuração (Correção)
        K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
        x_hat = x_hat + np.dot(K, (sample - np.dot(H, x_hat)))
        P = P - np.dot(np.dot(K, H), P)

        filtered_signal.append(x_hat[0])  # Apenas a primeira componente é o sinal estimado

    return np.array(filtered_signal)

def kalman(signal, window_size, order, sample_rate, SNR_dB=10.):
    filtered_signal = []

    for i in range(0, len(signal), window_size):
        window_samples = signal[i:i+window_size]
        
        # Realizar análise LPC e construir as matrizes Ak e H
        A, sigma, _ = lpc_analysis(window_samples, order)
        Ak, H = build_matrices(A, len(window_samples))

        # Calcular a variância do erro de aquisição R com base no SNR linear
        SNR_linear = 10.0**(SNR_dB / 10.0)
        Rx = 1.0 / SNR_linear
        
        # Calcular Q e R (assumindo que não mudam dentro da janela)
        Q = np.eye(order) * sigma  # Variância do erro de predição
        R = np.eye(1) * Rx  # Variância do erro de aquisição

        # Aplicar o filtro de Kalman na janela
        filtered_window = kalman_filter(window_samples, Ak, H, Q, R)
        filtered_signal.extend(filtered_window)

    return np.array(filtered_signal)