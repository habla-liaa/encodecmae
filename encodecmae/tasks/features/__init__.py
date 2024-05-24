import soundfile as sf
import random
import numpy as np
from typing import Dict, Any, List, Union
from loguru import logger
import torchaudio
import torch

class Processor:
    """Base class for processors."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output.

        Args:
            x (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Processed data.
        """
        raise NotImplementedError

class SequentialProcessor(Processor):
    """Sequential processor that applies a list of processors sequentially."""

    def __init__(self, processors: List[Processor]) -> None:
        """Initialize SequentialProcessor.

        Args:
            processors (List[Processor]): List of processors.
        """
        self._processors = [p() for p in processors]
    
    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Process input by applying each processor sequentially.

        Args:
            x (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Processed data.
        """
        for p in self._processors:
            x = p(x)
        return x

class ReadAudioProcessor(Processor):
    """Processor to read audio files."""

    def __init__(self, key_in: str, key_out: str, max_length: Union[float, None] = None, mono: bool = True) -> None:
        """Initialize ReadAudioProcessor.

        Args:
            key_in (str): Key for input audio.
            key_out (str): Key for output audio.
            max_length (Union[float, None], optional): Maximum length of audio in seconds. Defaults to None.
            mono (bool, optional): Whether to convert stereo audio to mono. Defaults to True.
        """
        super().__init__()
        self.key_in, self.key_out, self.max_length, self.mono = key_in, key_out, max_length, mono

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Read audio file and process it.

        Args:
            x (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Processed data.
        """
        try:
            if self.max_length is not None:
                audio_info = sf.info(x[self.key_in])
                desired_frames = int(self.max_length*audio_info.samplerate)
                total_frames = audio_info.frames
                if total_frames > desired_frames:
                    start = random.randint(0,total_frames - desired_frames)
                    stop = start + desired_frames
                else:
                    start = 0
                    stop = None
            else:
                start = 0
                stop = None
            if 'start' in x:
                start = x['start']
            if 'stop' in x:
                stop = x['stop']
            x['start'] = start
            x['stop'] = stop
            wav, fs = sf.read(x[self.key_in], start=start, stop=stop, dtype=np.float32)
            if (wav.ndim == 2) and self.mono:
                wav = np.mean(wav,axis=-1)
        except Exception as e:
            logger.warning('Failed reading {}'.format(x[self.key_in]))
            wav = None
        x[self.key_out] = wav
        return x

class LoadNumpyProcessor(Processor):
    """Processor to load numpy arrays."""

    def __init__(self, key_in: str, key_out: str) -> None:
        """Initialize LoadNumpyProcessor.

        Args:
            key_in (str): Key for input numpy array file.
            key_out (str): Key for output numpy array.
        """
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Load numpy array and process it.

        Args:
            x (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Processed data.
        """
        x[self.key_out] = np.load(x[self.key_in])
        return x

class MelspectrogramProcessor(Processor):
    """Processor to calculate melspectrograms from waveforms.
       Internally, torchaudio.compliance.kaldi.fbank is used and the same kwargs are accepted.
       Additionally, norm_stats can be supplied with [mean,std] to normalize the resulting melspectrogram.
       key_in and key_out are strings indicating the key containing the waveform 
       and the key where the result will be stored."""

    def __init__(self, key_in='wav', key_out='wav_features',
                        frame_length=25,
                        frame_shift=10, 
                        high_freq=0, 
                        htk_compat=False, 
                        low_freq=20,
                        num_mel_bins=23,
                        sample_frequency=16000,
                        window_type='povey',
                        dither=0.0,
                        use_energy=False, norm_stats=[0,1]):
        super().__init__()
        self.mel_kwargs = dict(frame_length=frame_length,
                        frame_shift=frame_shift, high_freq=high_freq,
                        htk_compat=htk_compat, low_freq=low_freq, num_mel_bins=num_mel_bins,
                        sample_frequency=sample_frequency, window_type=window_type, use_energy=use_energy)
        self.norm_stats = norm_stats
        self.key_in, self.key_out = key_in, key_out

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate melspectrogram and normalize it.

        Args:
            x (Dict[str, Any]): Input data dictionary containing waveform.

        Returns:
            Dict[str, Any]: Output data dictionary containing melspectrogram.

        """
        if x[self.key_in] is not None:
            mel = torchaudio.compliance.kaldi.fbank(torch.from_numpy(x[self.key_in]).unsqueeze(0), **self.mel_kwargs).numpy()
            mel = (mel-self.norm_stats[0])/self.norm_stats[1]
        else:
            mel = None
        x[self.key_out] = mel
        return x

class SpectrogramProcessor(Processor):
    """Processor to calculate spectrograms from waveforms.
    Internally, torchaudio.compliance.kaldi.spectrogram is used and the same kwargs are accepted.
    Additionally, norm_stats can be supplied with [mean,std] to normalize the resulting spectrogram.
    key_in and key_out are strings indicating the key containing the waveform 
    and the key where the result will be stored."""
    def __init__(self, key_in='wav', key_out='wav_features',
                       frame_length=25, 
                        frame_shift=10,
                        sample_frequency=16000,
                        window_type='povey',
                        dither=0.0,
                        norm_stats=[0,1]):
        self.spec_kwargs = dict(frame_length=frame_length,
                      frame_shift=frame_shift,
                      sample_frequency=sample_frequency, 
                      window_type=window_type, 
                      dither=dither)
        self.norm_stats = norm_stats
        self.key_in, self.key_out = key_in, key_out    

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate spectrogram and normalize it.

        Args:
            x (Dict[str, Any]): Input data dictionary containing waveform.

        Returns:
            Dict[str, Any]: Output data dictionary containing spectrogram.

        """
        spec = torchaudio.compliance.kaldi.spectrogram(torch.from_numpy(x[self.key_in]).unsqueeze(0), **self.spec_kwargs).numpy()
        spec = (spec-self.norm_stats[0])/self.norm_stats[1]
        x[self.key_out] = spec
        return x
