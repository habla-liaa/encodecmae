import soundfile as sf
import random
import numpy as np

def ProcessorReadAudio(x,
                       state,
                       input=None,
                       output=None,
                       max_length=None,
                       mono=True
):  
    def read_sample(x,state,max_length,mono):
        if max_length is not None:
            audio_info = sf.info(x[input])
            desired_frames = int(max_length*audio_info.samplerate)
            total_frames = audio_info.frames
            if total_frames > desired_frames:
                start = random.randint(0,total_frames - desired_frames)
                stop = start + desired_frames
            else:
                start = 0
                stop = None
            if 'chunk_idx' in state:
                #This is for ordered reading in chunks when doing evals
                start = int(state['chunk_idx']*desired_frames)
                stop = start + desired_frames
        else:
            start = 0
            stop = None
        if 'start' in x:
            start = x['start']
        if 'stop' in x:
            stop = x['stop']
        x['start'] = start
        x['stop'] = stop
        wav, fs = sf.read(x[input], start=start, stop=stop, dtype=np.float32)
        if (wav.ndim == 2) and mono:
            wav = np.mean(wav,axis=-1)
        return wav
    try:
        wav = read_sample(x, state, max_length, mono)
    except:
        print('Failed reading {}'.format(x))
        wav = None
    if output is None:
        output = input
    x[output] = wav
    
    return x, state

def ProcessorLoadNumpy(x, state, input, output):
    x[output] = np.load(x[input])
    return x,state