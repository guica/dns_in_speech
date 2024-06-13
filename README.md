# Noise Removal in Voice Signals: Deep Neural Networks vs. Wiener and Kalman Filters

## Introduction
This repository contains the code and resources for the thesis "Noise Removal in Voice Signals: Deep Neural Networks vs. Wiener and Kalman Filters," presented by Guilherme de Camargo as part of his Electrical Engineering degree at Universidade Presbiteriana Mackenzie. The project compares traditional noise removal methods with deep learning approaches to improve voice signal quality and intelligibility.

## Objectives
### General Objective
- Compare the quality and performance of noise removal in voice signals using deep neural networks and traditional filters (Wiener and Kalman).

### Specific Objectives
- Implement traditional filters and evaluate their performance and quality in noise removal.
- Compare results in terms of perceptual quality, processing time, and computational resources.

## Methodology
### Data
- Voice Data: Clean voice signals collected from various sources such as SPOLTECH and Common Voice.
- Noise Data: Diverse noise types, including stationary and non-stationary noises from datasets like ESC-50 and CHiME-6.

### Traditional Filters
- **Wiener Filter**: Applied based on statistical properties of the signal and noise.
- **Kalman Filter**: Utilizes dynamic models to estimate the voice and noise components using autoregressive model.

### Deep Neural Networks
- **Attention Res U-Net**: An advanced neural network model incorporating attention mechanisms and residual connections [available here](https://ieeexplore.ieee.org/document/9902215.).
- **FCRN with Non-Intrusive PESQNet**: Fully Convolutional Recurrent Network mediated by a non-intrusive PESQNet [available here](https://ieeexplore.ieee.org/document/9750869.).
- **PRIDNet**: Pyramid Real Image Denoising Network with quantization techniques for reduced computational complexity [available here](https://ieeexplore.ieee.org/document/9750869.) and [here](https://arxiv.org/pdf/1808.06474.pdf.).

## Implementation
### Environment
- Programming Language: Python
- Libraries: TensorFlow, Keras, NumPy, SciPy, Librosa.

### Setup
1. Clone the repository:
    ```sh
    git clone https://github.com/guica/dns_in_speech.git
    cd dns_in_speech
    ```
2. Docker:
    ```sh
    sudo docker compose up
    ```
    
## Results
The study demonstrates that deep neural network models, particularly the Attention Res U-Net and PRIDNet, significantly outperform traditional filters in terms of perceptual quality and intelligibility of voice signals. However, these models require higher computational resources. The quantization of PRIDNet reduces computational complexity while maintaining  performance.

### Comparative Analysis
- **SNR Improvement**: All models were evaluated based on the improvement in Signal-to-Noise Ratio (SNR).
- **PESQ Scores**: Perceptual Evaluation of Speech Quality (PESQ) was used to measure the quality of the enhanced signals.
- **STOI Scores**: Short-Time Objective Intelligibility (STOI) provided insights into the intelligibility of the processed signals.

## Conclusion
Deep neural networks offer superior performance in noise removal for voice signals compared to traditional methods. While computational demands are higher, techniques like model quantization can make these advanced models feasible for real-time applications in lower-end devices.

## Acknowledgments
Special thanks to Prof. Dr. Talitha Nicoletti Régis for her guidance, and to Universidade Presbiteriana Mackenzie for the support.

## References
- Xu et al., 2022
- Fu et al., 2019
- Meyer et al., 2020
- Bulut and Koishida, 2020
- Tu et al., 2020

For more detailed information, please refer to the [thesis document](https://drive.google.com/file/d/1Rip2yPE2Kr3QKVoIGdPtJprGVsr7Xggv/view?usp=sharing).
