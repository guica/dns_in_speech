import numpy as np

def bss_eval_sources(se, s):
    """
    Avalia a qualidade da separação de um sinal de áudio estimado (se) em relação a um sinal de áudio verdadeiro (s).
    
    Args:
        se (numpy.ndarray): Sinal de áudio estimado (1D array).
        s (numpy.ndarray): Sinal de áudio verdadeiro (1D array).

    Returns:
        float: Valor SDR (Signal-to-Distortion Ratio).
        float: Valor SIR (Signal-to-Interference Ratio).
        float: Valor SAR (Signal-to-Artifact Ratio).
    """
    if len(se) != len(s):
        raise ValueError("Os sinais estimados e verdadeiros devem ter o mesmo comprimento.")

    # Cálculo do SDR
    signal_power = np.sum(s ** 2)
    distortion_power = np.sum((s - se) ** 2)
    SDR = 10 * np.log10(signal_power / distortion_power)

    # Cálculo do SIR
    interference_power = np.sum((s - se) ** 2) - distortion_power
    SIR = 10 * np.log10(signal_power / interference_power)

    # Cálculo do SAR
    artifact_power = np.sum(se ** 2) - distortion_power
    SAR = 10 * np.log10(signal_power / artifact_power)

    return SDR, SIR, SAR