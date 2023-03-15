# Official implementation of Sec2Sec Co-attention Transformer

Abstract: Emotion is associated with thoughts, feelings and a degree of pleasure, which can accordingly affect people's decision-making. Two major dimensions of emotion - valence and arousal have been widely studied in the literature. Researchers and practitioners have developed many methods to detect emotions from textual data. Video-based emotion detection is less explored due to some challenges. For example, how to represent different modalities (visual and audio) in a video, how to capture their alignment, and how to model the temporal structure can significantly affect the model performance. In this study, we develop a novel LSTM-based network with a Transformer co-attention mechanism to predict video emotions. Results of experiments show that our proposed model outperforms several state-of-the-art baselines by a large margin. In addition, we also conduct extensive data analysis to understand how different dimensions of visual and audio components are related to video emotions. Our model has some interpretability, allowing us to investigate the role of different timepoints in the overall video emotion, which has practical implications for video designers. 

### Dependencies:
- CUDA 11.7
- PyTorch 1.9.1
