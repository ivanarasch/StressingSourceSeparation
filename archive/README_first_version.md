# How do State-of-the-Art Source Separation Models Perform with Challenging Audio Modifications Compared to Unprocessed Data?

## Table of contents

1. Introduction
2. Literature review
3. Pipeline
4. Dataset and architecture
5. Models and methods
6. Experiments, preprocessing
7. Presentation of evaluation metrics
8. Analysis and discussion
9. Conclusion
10. Bibliography

***

## 1. Introduction, background and overview of project

Through our project, we aim to answer the following question:

#### How do state-of-the-art methods handle challenging conditions?

We used the MUSDB18 dataset and the source separation models Spleeter and Demucs. Source separation reverses the mixing process to isolate stems. Models learn patterns to reconstruct isolated sources from a full mix.

We focused on two factors:

- Impact of different non-vocal elements on extracting vocals
- Impact of different audio processing effects on separation of all sources

## 2. Literature review
Through a review of literature on BSS, MUSDB18, Spleeter, Demucs, and evaluation metrics, limitations and strengths were identified.

MUSDB18 and the chosen models are industry standard. Current BSS struggles with limited datasets, mainstream genres, and phase handling. Wave-U-Net (used in Demucs) improves time-domain separation. Evaluation metrics often rely on ground truth stems, though embedding-based metrics offer perceptual improvements.

In conclusion, BSS struggles with unusual conditions. Our experiments address interference from non-vocal elements and effects processing, filling gaps in the literature.

## 3. Pipeline

We chose the following pipeline for our project for each experiment:

#### Experiment 1: Impact of instrumental interference on isolation of vocal tracks when processed using Spleeter and Demucs

1. Obtain MUSDB18 tracks (50 test tracks)
2. Extract stems
3. Combine vocals with each non-vocal stem
4. Compile processed data
5. Run through Spleeter and Demucs
6. Obtain separated sources
7. Evaluate using standard metrics
8. Analyze results

#### Experiment 2: Impact of common audio effects on good separation of effected vocals from all non-vocal stems 

1. Obtain MUSDB18 tracks (50 test tracks)
2. Extract stems
3. Apply effects to vocals (reverb, delay, bitcrush, compression)
4. Combine affected vocals with remaining stems
5. Compile processed data
6. Run through Spleeter and Demucs
7. Obtain separated sources
8. Evaluate and analyze

## 4. Dataset and architecture

We used MUSDB18: 150 tracks (100 train, 50 test), stereo, 44.1 kHz, 256 kbps. Stems: vocals, bass, drums, accompaniment, other.

#### Experiment 1: Impact of instrumental interference on isolation of vocal tracks when processed using Spleeter and Demucs

- musdb18SynthesizedExperiment1/
	- TrackName1/
		- accompainment_vocals.wav
        - bass_vocals.wav
        - drums_vocals.wav
        - other_vocals.wav

And so on for each track. Total of 200 songs

#### Experiment 2: Impact of common audio effects on good separation of effected vocals from all non-vocal stems 

- musdb18SynthesizedExperiment2/
	- TrackName1/
        - TrackName1_original.wav
        - TrackName1_delay_l1.wav
        - TrackName1_delay_l2.wav
        - TrackName1_delay_l4.wav
        - TrackName1_bitcrush_l1.wav
        - TrackName1_bitcrush_l2.wav
        - TrackName1_bitcrush_l4.wav
        -  TrackName1_compression_l1.wav
        -  TrackName1_compression_l2.wav
        -  TrackName1_compression_l4.wav
        -  TrackName1_reverb_l1.wav
        -  TrackName1_reverb_l2.wav
        -  TrackName1_reverb_l4.wav
	
And so on for each track. Total of 650 songs

## 5. Models

For this study, we compared source separation performances using Spleeter and Demucs.

#### Spleeter:

Spleeter is an open-source music source separation library developed by Deezer in 2019. It separates audio into vocals, drums, bass, and accompaniment using a U-Net-based neural network. Pre-trained models include 2-, 4-, and 5-stem separations. It runs on GPU, with results comparable to Demucs. For Experiment 1, we used the 2-stem model.

Pros:
- Fast processing of audio files
- Adaptable to multiple uses

Cons:
- Diminished performance on complex inputs
- Lacks flexibility, does not retrain itself, likely struggling with vocals from elements with similar harmonic content (guitar) and processed inputs in Experiment 2

#### Demucs:

Demucs is an open-source music source separation model developed by Facebook AI Research in 2019. It operates directly on the raw audio waveform using a Wave-U-Net architecture, learning temporal and spectral features more intelligently. It separates audio into multiple stems, including vocals, drums, bass, and other accompaniment, and can run on a GPU.

Pros:
- High-quality output with fewer artifacts than spectrogram-based methods
- Handles complex input and overlapping harmonic structures
- Flexible architecture allows fine-tuning for different instruments or genres

Cons:
- Computationally inefficient and costly
- Implementation not user friendly

These remarks suggest Demucs may perform well in separating vocals from elements with similar harmonic content in Experiment 1 and with processed synthesized inputs in Experiment 2.

## 6. Experiments and preprocessing
Though referred to as experiments, carrying out the experiment itself is fairly simple: feed the preprocessed audio into Spleeter or Demucs, run the separation, and collect the outputs. The bulk of our work was done through the data synthesis. We synthesized the 50 test tracks included in MUSDB18.

We made use of the ffmpeg library as a tool for audio processing, it allowed for the synthesis of data for the purpose of our experiments. It allowed for the application of effects (experiment 2) as well as the compilation of different stems together (Experiments 1 and 2). It was efficient in the preparation of our large datasets and allowed for them to be ready to use with the models.

Experiment 1: Impact of instrumental interference on isolation of vocal tracks when processed using Spleeter and Demucs

As previously mentioned, through our study, we wished to understand the impact that different instruments and accompaniments can have on the separation of vocals. For each track of the dataset, we generated new audio by pairing the vocal stem with each instrument stem, allowing us to evaluate how well the models could separate them.

This aligns with common practices within MIR and audio engineering research.

To create this new data, we combined the isolated vocal stem with each individual non-vocal stem (drums, bass, other, and accompaniment) from the MUSDB18 dataset using Python and the musdb library.

We iterated over each track to extract the original stems and then created new synthetic mixtures by adding vocals to a single stem. We then wrote each mixture as a WAV file in a structured directory alongside the original stems.

See examples below:
```
# Libraries used:
import musdb
import stempeg
import ffmpeg
import utils
import importlib
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import IPython.display as ipd
from pathlib import Path

# Data synthesis for experiment 1

vocals = track.targets['vocals'].audio
drums = track.targets['drums'].audio
accompaniment = track.targets ['accompaniment'].audio
bass = track.targets['bass'].audio
other = track.targets['other'].audio

drums_vocals = drums + vocals
accompaniment_vocals = accompaniment + vocals
bass_vocals = bass + vocals
other_vocals = other + vocals

import soundfile as sf

sf.write("drums_vocals.wav", drums_vocals, track.rate)
sf.write("accompaniment_vocals.wav", accompaniment_vocals, track.rate)
sf.write("bass_vocals.wav", bass_vocals, track.rate)
sf.write("other_vocals.wav", other_vocals, track.rate)
```
  
Once we had synthesized the new data, as outlined in the last block of code seen above, we were able to use it for source separation.

#### Experiment 2: Impact of common audio effects on good separation of effected vocals from all non-vocal stems 

The next exploration was the models’ performance on stems processed with common audio effects: reverb, delay, bitcrush, and compression, applied at different intensities to test when models fail to separate the processed stem without affecting others.

This is relevant to MIR and audio engineering, as current models struggle with overlapping instruments or applied effects.

Using Python, effects were applied to the vocal stem of each MUSDB18 track at 3 intensity levels. Reverb and delay varied in time and repeats, bitcrush in bit depth and downsampling, and compression in threshold and ratio. Each processed vocal replaced the original stem, keeping other stems unchanged, and resulting mixes were saved as .wav files.

Future work could explore effects on other stems.

The method used each effect in separate loops manually, which works, but is repetitive, harder to maintain, and less flexible. Future work could explore modular effect chains or object-oriented classes to reduce duplication and improve flexibility. Refer to DataSynthesis.ipynb for all code used in this process.

## 7. Presentation of evaluation metrics

To evaluate source separation models, standard metrics such as SAR, SDR, SIR, and ISR are used. SDR measures overall quality, SIR quantifies unwanted leakage, SAR evaluates artifacts, and ISR assesses stereo/spatial accuracy.

These metrics have limitations: high scores may not reflect human perception, and they rely on ground truth stems, which are often unavailable. WASPAA proposed embedding-based metrics for perceptual evaluation without ground truth, but these remain largely inaccessible. Museval, previously standard, is deprecated.

We used SI-SDR and SI-SAR as evaluation metrics. They improve on standard metrics, remain simple to apply, and allow us to objectively assess model performance on the MUSDB18 dataset under challenging conditions.

## 8. Analysis and discussion

#### Experiment 1 scores and evaluation

Using the separated stems in the MUSDB18 dataset, we evaluated the performance of Spleeter and Demucs with the help of SI-SDR, SI-SAR-like, and RMSE metrics. These  stems served as their own control data upon whcih to evaluate performance, allowing us to assess how different non-vocal elements interfere with vocal separation.

Across both models, instruments that overlap heavily in frequency or harmonic content with the vocal stem caused the most interference when loooking at separated vocals. Both Spleeter and Demucs performed well, resulting in high separation scores for both metrics, on bass and drums. These instrument categories can be noted as not having much harmonic overlap with the spectrum of a vocal. The “other” and "accompaniment" categories, generally guitar and piano recordings respectively, resulted in the lowest performance scores, indicating more interference. 

The trend demonstrated by both models was that densely textured or harmonically rich stems (e.g. guitar and piano recordings) highly affected vocal separation performance, whereas percussive or tonally distinct instruments (drums, bass) had a smaller impact. However, #### Demucs outperformed #### Spleeter on every stem pairing, achieving higher SI-SDR and SI-SAR-like values and lower RMSE than Spleeter for all vocal and non-vocal mixtures. 

For each instrument, the following scores were found:

Bass: 

Spleeter
- SI-SDR = 15.04
- SI-SAR-like = 15.18
- RMSE = 0.00868

Demucs
- SI-SDR = 20.94
- SI-SAR-like = 20.98
- RMSE = 0.00459

Drums: 

Spleeter
- SI-SDR = 11.78
- SI-SAR-like = 12.05
- RMSE = 0.01154

Demucs
- SI-SDR = 15.57
- SI-SAR-like = 15.69
- RMSE = 0.00762

Other: 

Spleeter
- SI-SDR = 7.86
- SI-SAR-like = 8.52
- RMSE = 0.01804

Demucs
- SI-SDR = 9.84
- SI-SAR-like = 10.27
- RMSE = 0.01384

Accompaniment: 

Spleeter
- SI-SDR = 6.50
- SI-SAR-like = 7.38
- RMSE = 0.02124

Demucs
- SI-SDR = 8.75
- SI-SAR-like = 9.29
- RMSE = 0.01576

These findings support our claim that both models perform well when separating vocals from dums and bass, with Demucs outperforming Spleeter, and that both perform poorly in spearting vocals from accompaniment and other.

#### Experiment 2 scores and evaluation

Using the the unprocessed stems provided in the MUSDB18 dataset, we evaluated the performance of Spleeter and Demucs with the help of SI-SDR, SI-SAR-like, RMSE metrics. These unprocessed stems served as control data to compare with and upon which we could evaluate the capabilities of these models.

Across both models, effects that extend or distort time-domain structure (delay, reverb) caused the most damage. Effects that modify spectral resolution (bitcrush) or dynamic range (compression) were less harmful unless pushed to extremes.

The trend demonstrated by both models was that reverb highly effected the performance of both models, especially when it was applied to its highest level. The same was found for delay. Comppression caused moderate impact in level 4, resulting in major artifacts, while bitcrushing was the least disruptive. 

For each model, the following  worst scores were found:
#### Spleeter:
Reverb
- Vocal SI-SDR (Level 4): –20.5 dB
- Other SI-SDR (Level 4, average): 1.9 dB
- RMSE (vocal, Level 4): 0.052
  
Delay
- Vocal SI-SDR (Level 4): 2.7 dB
- RMSE (vocal, Level 4): 0.041
  
Compression
- Vocal SI-SDR (Level 4): 4.0 dB
- RMSE (vocal, Level 4): 0.033
  
Bitcrushing
- Vocal SI-SDR (all levels): 7.6 dB
- RMSE (vocal, all levels): 0.028

#### Demucs:
Reverb
- Vocal SI-SDR (Level 4): –19.99 dB
- Other SI-SDR (Level 4, average): 1.53–4.08 dB
- RMSE (vocal, Level 4): 0.1226
  
Delay
- Vocal SI-SDR (Level 4): 2.51 dB
- RMSE (vocal, Level 4): 0.0346
  
Compression
- Vocal SI-SDR (Level 4): 1.19 dB
- RMSE (vocal, Level 4): 0.0348
  
Bitcrushing
- Vocal SI-SDR (all levels): 8.7–9.1 dB
- RMSE (vocal, all levels): 0.0128–0.0136

From these results, we can see that both models fail badly at high levels of reverb, with Spleeter having slightly lower RMSE. The same is seen for delay, however in this instance Demucs presents the lower RMSE. In compressing, Spleeter outperforms Demucs overall, and in bitcrushing Demucs outperforms Spleeter. Thus, it is difficult to state which model performed better overall. What can be reiterated is that both models, at high levels of processing, source seperation is poorly accomplished.

More specifically, Spleeter shows stronger performance when dynamics are distorted, such as in compression, while Demucs performs better when distorttion is applied in bitcrushing. When exaggerated temporal smearing is introduced with reverb and delay, again both models fail. However their failures differ: Spleeter shows better reconstruction of waveforms, whereas Demucs produces more artifacts in the reconstructed waveform.

## 9. Conclusion

In conclusion, the first experiment showed that non-vocal elements with similar harmonic content to vocals cause the most interference, while drums and bass allow better separation. The second experiment showed both models struggle with highly processed vocals, especially with reverb and delay, while compression and bitcrushing result in better separation. Models also occasionally misassign effects to non-vocal elements. 

Future work could include training on stems with effects, expanding datasets with more instruments and complex arrangements, and using multi-channel or advanced waveform modeling to improve separation.

One challenge to retain from the second experiment is that the model struggled in keeping the effects only tied to the vocal stems, occasionally erroniously separating the effects with non-vocal elements. This is an audible observation that leaves us with the opportunity to further explore how these models can better distinguish assign applied effects to the correct sources, ensuring that reverberation, delay, or other processing remains confined to the intended stem without leaking into other separated tracks.

Overall, it is impossible to definitively say which model performs better under challenging conditions, however Demucs showed marginal superiority.

## 10. Bibliography

Bereuter, F., Stahl, J., Plumbley, M., & Sontacchi, J. (2025). Musical source separation bake-off: Comparing objective metrics with human perception. [Manuscript in preparation].

De Pra, Y., & Fontana, F. (2018). Development of real‑time audio applications using Python. IRIS — University of Udine. https://hdl.handle.net/11390/1147041

Defossez, A., Usunier, N., Bottou, L., & Jégou, H. (2019). Two-step sound source separation: Training on learned latent targets. arXiv. https://arxiv.org/abs/1910.09804

Hennequin, R., Khlif, A., Voituret, F., & Moussallam, M. (2020). Spleeter: a fast and efficient music source separation tool with pre-trained models. Journal of Open Source Software, 5(50), 2154.

Le Roux, J., Wisdom, S., Hennequin, R., & Vincent, E. (2019). SDR — Half-baked or well done? Google Research. https://research.google/pubs/sdr-half-baked-or-well-done/

McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. In B. V. Das & S. R. (Eds.), Proceedings of the 14th Python in Science Conference (pp. 18–25). https://librosa.org

Namballa, R., Roginska, A., & Fuentes, J. (2025). Do music source separation models preserve spatial information in binaural audio? [Conference paper]. https://www.researchgate.net/publication/393260674_Do_Music_Source_Separation_Models_Preserve_Spatial_Information_in_Binaural_Audio

Rafii, Z., Liutkus, A., Stöter, F.-R., Mimilakis, S., Bittner, R., & others. (2017). musdb18: A corpus for music separation. https://sigsep.github.io/musdb/

Stoller, D., Ewert, S., & Dixon, S. (2018). Wave-U-Net: A multi-scale neural network for end-to-end audio source separation. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR).

Vincent, E., Gribonval, R., & Févotte, C. (2006). Performance measurement in blind audio source separation. IEEE Transactions on Audio, Speech, and Language Processing, 14(4), 1462–1469. https://doi.org/10.1109/TSA.2005.858005
