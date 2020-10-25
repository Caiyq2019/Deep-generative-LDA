# Deep-LDA
A Pytorch implementations of a deep generative version of LDA for author's article ["Deep generative LDA"](https://arxiv.org/)

Code inherited from ["DNF"] (https://github.com/Caiyq2019/Deep-normalization-for-speaker-vectors)

## Datasets
```bash
trainingset:Voxceleb 
testset: SITW, CNCeleb
```
Following this [link](https://pan.baidu.com/s/1NZXZhKbrJUk75FDD4_p6PQ) to download the dataset 
(extraction code：8xwe)

## Run DNF
```bash
python train.py
```
The evaluation and scoring will be performed automatically during the training process.

## Other instructions
```bash
score.py is a python implementations of the standard kaldi consine scoring, you can also use kaldi to do the plda scoring
tsne.py can be used to draw the distribution of latent space 
```


