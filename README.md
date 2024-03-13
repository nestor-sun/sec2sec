# Official implementation of Sec2Sec Co-attention Transformer

Abstract: Emotion is associated with thoughts, feelings and a degree of pleasure, which can accordingly affect people's decision-making. Two major dimensions of emotion - valence and arousal have been widely studied in the literature. Researchers and practitioners have developed many methods to detect emotions from textual data. Video-based emotion detection is less explored due to some challenges. For example, how to represent different modalities (visual and audio) in a video, how to capture their alignment, and how to model the temporal structure can significantly affect the model performance. In this study, we develop a novel LSTM-based network with a Transformer co-attention mechanism to predict video emotions. Results of experiments show that our proposed model outperforms several state-of-the-art baselines by a large margin. In addition, we also conduct extensive data analysis to understand how different dimensions of visual and audio components are related to video emotions. Our model has some interpretability, allowing us to investigate the role of different timepoints in the overall video emotion, which has practical implications for video designers. 

## Model Architecture
![sec2sec framework](https://user-images.githubusercontent.com/47902113/225509553-f2c3bada-5691-4c97-a855-73aea9f702f6.png)

## What's New
**[Mar 2023]:** Sec2Sec Co-attention Transformer released.

**[Dec 2023]:** Sec2Sec Co-attention Transformer was accepted as a regular paper at International Conference in Acoustics, Speech and Signal Processing (ICASSP).

## How to run
- [dataloader.py](dataloader.py): load LIRIS audio-visual data
- [layer.py](layer.py): implementation of our model
- [train.py](train.py): train and test our model
- [utils.py](utils.py): necessary functions for training our model

### Dependencies:
- Python 3.8.8
- CUDA 11.7
- PyTorch 1.9.1
