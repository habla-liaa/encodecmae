<h2 align="center">EnCodecMAE: Leveraging neural codecs for universal audio representation learning</h2>

<p align="center">
    <a href="http://arxiv.org/abs/2309.07391">
        <img alt="read the paper" src="https://img.shields.io/badge/Read_the_paper-2ea44f">
    </a>
    <a href="https://colab.research.google.com/drive/123Zn6h0DRVcjsLFp8Xl4j0PZlZ-7VsK2?usp=sharing">
        <img alt="run in colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://"><img src="https://img.shields.io/badge/Cite-Bibtex-2ea44f" alt="Cite - Bibtex"></a>
</p>

This is EnCodecMAE, an audio feature extractor pretrained with masked language modelling to predict discrete targets generated by EnCodec, a neural audio codec. 
For more details about the architecture and pretraining procedure, read the [paper](https://arxiv.org/abs/2309.07391).

## Updates:
- 2024/5/23 Updated paper in arxiv. New models with better performance across all downstream tasks are available for feature extraction. Code for older version is [here](https://github.com/habla-liaa/encodecmae/tree/v.1.0.0)
- 2024/2/29 [New code](https://github.com/mrpep/encodecmae-to-wav) to go from encodecmae to the waveform domain, with pretrained generative audio models from [this paper](https://arxiv.org/abs/2402.09318).
- 2024/2/14 [Leveraging Pre-Trained Autoencoders for Interpretable Prototype Learning of Music Audio](https://arxiv.org/abs/2402.09318) was accepted to ICASSP 2024 XAI Workshop.
- 2023/10/23 [Prompting for audio generation](https://mrpep.github.io/myblog/posts/audio-lm/).

## Usage

### Feature extraction using pretrained models

#### Try our example [Colab notebook](https://colab.research.google.com/drive/123Zn6h0DRVcjsLFp8Xl4j0PZlZ-7VsK2?usp=sharing) or

#### 1) Clone the [EnCodecMAE library](https://github.com/habla-liaa/encodecmae):
```
git clone https://github.com/habla-liaa/encodecmae.git
```

#### 2) Install it:

```
cd encodecmae
pip install -e .
```

#### 3) Extract embeddings in Python:

``` python
from encodecmae import load_model

model = load_model('mel256-ec-base_st', device='cuda:0')
features = model.extract_features_from_file('gsc/bed/00176480_nohash_0.wav')
```

### Pretrain your models

#### 1) Install docker and docker-compose in your system. You'll also need to install nvidia-container toolkit to access GPUs from a docker container.
#### 2) Execute the start_docker.sh script

First, docker-compose.yml has to be modified. In the volumes section, change the routes to the ones in your system. You'll need a folder called datasets with the following subfolders:
- audioset_24k/unbalanced_train
- fma_large_24k
- librilight_med_24k

All the audio files need to be converted to a 24kHz sampling rate.

You might also modify the device_ids if you have a different number of gpus.

Then, run:
```
chmod +x start_docker.sh
./start_docker.sh
```
This will build the encodecmae image, start a container using docker compose, and attach to it.

#### 3) Install the encodecmae package inside the container
```
cd workspace/encodecmae
pip install -e .
```

#### 4) Run the training script
```
chmod +x scripts/run_pretraining.sh
scripts/run_pretraining.sh
```

The training script uses my own library for executing pipelines configured with gin: ginpipe. By modifying the config files (with .gin extension), you can control aspects of the training and the model configuration. I plan to explain my approach to ML pipelines, and how to use gin and ginpipe in a future blog article. Stay tuned!
