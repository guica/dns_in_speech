import numpy as np
from scipy.signal import wiener

def wiener_filter(noisy_signal, snr_db):
    """
    Aplica o filtro de Wiener para remover ruído de um sinal ruidoso.

    Args:
        noisy_signal (numpy array): O sinal de voz ruidoso.
        snr_db (float): Relação sinal-ruído (SNR) em dB presente no sinal de voz ruidoso.

    Returns:
        filtered_signal (numpy array): O sinal filtrado.
    """
    # Calcule a potência do sinal ruidoso
    signal_power = np.mean(noisy_signal ** 2)

    # Calcule a potência do ruído com base na relação sinal-ruído desejada em dB
    noise_power = signal_power / (1 + (10 ** (snr_db / 10)))

    filtered_signal = wiener(noisy_signal.astype('float32'), mysize=29, noise=noise_power)

    return filtered_signal