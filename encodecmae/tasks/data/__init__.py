import pandas as pd
from loguru import logger
from pathlib import Path
import numpy as np
from tqdm import tqdm
import soundfile as sf
from torch.utils.data import Dataset
import copy
import torch
import re
import joblib

def load_dataset(state, reader_fn, 
                 cache=True, 
                 filters=[], 
                 key_out='dataset_metadata',
                 rename=None):
    
    if not (cache and key_out in state):
        if not isinstance(reader_fn, list):
            reader_fn = [reader_fn]
        dfs = [fn() for fn in reader_fn]
        df = pd.concat(dfs).reset_index()
        state[key_out] = df
    else:
        logger.info('Caching dataset metadata from state')
    
    for f in filters:
        state[key_out] = f(state[key_out])
    if rename is not None:
        for r in rename:
            state[key_out][r['column']] = state[key_out][r['column']].apply(lambda x: r['new_value'] if x == r['value'] else x)
    
    return state

def read_audiodir(dataset_path, subsample=None, dataset=None, regex_groups=None, filter_list=None, partition_lists=None,filter_mode='include'):
    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]
    all_files = []
    for p in dataset_path:
        all_files_i = list(Path(p).rglob('*.wav')) + list(Path(p).rglob('*.flac'))
        all_files.extend(all_files_i)
    if filter_list is not None:
        with open(filter_list, 'r') as f:
            keep_values = set(f.read().splitlines())
        n_slashes = len(next(iter(keep_values)).split('/')) - 1
        stem_to_f = {'/'.join(v.parts[-n_slashes-1:]): v for v in all_files}
        if filter_mode == 'include':
            all_files = [stem_to_f[k] for k in keep_values]
        elif filter_mode == 'discard':
            all_files = [v for k,v in stem_to_f.items() if k not in keep_values]
        else:
            raise Exception("Unrecognized filter_mode {}".format(filter_mode))
    rows = []
    if subsample is not None:
        subsample_idx = np.random.choice(np.arange(len(all_files)),size=subsample,replace=False)
        all_files = np.array(all_files)[subsample_idx]
    for f in tqdm(all_files):
        try:
            finfo = sf.info(f)
            metadata = {'filename': str(f.resolve()),
                    'sr': finfo.samplerate,
                    'channels': finfo.channels,
                    'frames': finfo.frames,
                    'duration': finfo.duration}
            
            if regex_groups is not None:
                regex_data = re.match(regex_groups,str(f.relative_to(dataset_path[0]))).groupdict()
                metadata.update(regex_data)
            rows.append(metadata)
        except Exception as e:
            print(f'Failed reading {f}. {e}')
    df = pd.DataFrame(rows)
    if dataset is not None:
        df['dataset'] = dataset
    df['rel_path'] = df['filename'].apply(lambda x: str(Path(x).relative_to(dataset_path[0])))
    if partition_lists is not None:
        remainder = None
        map_to_partitions={}
        for k,v in partition_lists.items():
            if v is not None:
                list_path = Path(dataset_path[0],v)
                with open(list_path,'r') as f:
                    list_files = f.read().splitlines()
                for l in list_files:
                    map_to_partitions[str(l)] = k
            else:
                remainder = k
        df['partition'] = df['rel_path'].apply(lambda x: map_to_partitions[x] if x in map_to_partitions else remainder)
        df = df.drop('rel_path', axis=1)
    return df

def get_dataloaders(state, split_function=None, 
                           dataset_cls=None, 
                           dataloader_cls=None, 
                           dataset_key_in='dataset_metadata',
                           dataset_key_out='datasets',
                           partitions_key_out='partitions',
                           dataloaders_key_out='dataloaders'):

    if split_function is not None:
        partitions = split_function(state[dataset_key_in])
    else:
        partitions = {'train': state[dataset_key_in]}

    datasets = {k: dataset_cls[k](v, state) for k,v in partitions.items() if k in dataset_cls}
    dataloaders = {k: dataloader_cls[k](v) for k,v in datasets.items() if k in dataloader_cls}

    state[partitions_key_out] = partitions
    state[dataset_key_out] = datasets
    state[dataloaders_key_out] = dataloaders

    return state
    
def dataset_random_split(df, proportions={}):
    idxs = df.index
    prop_type = [v for k,v in proportions.items() if v>1]
    if len(prop_type)>0:
        prop_type = 'n'
    else:
        prop_type = 'prop'
    remainder_k = [k for k,v in proportions.items() if v==-1]
    if len(remainder_k) > 1:
        raise Exception("-1 can't be used in more than one entry")
    elif len(remainder_k) == 1:
        remainder_k = remainder_k[0]
    else:
        remainder_k = None
    partitions = {}
    for k,v in proportions.items():
        if k != remainder_k:
            if prop_type == 'prop':
                v = int(len(df)*v)
            sampled_idxs = np.random.choice(idxs, v, replace=False)
            idxs = [i for i in idxs if i not in sampled_idxs]
            partitions[k] = df.loc[sampled_idxs]
    if remainder_k is not None:
        partitions[remainder_k] = df.loc[idxs]
    return partitions
    
def remove_long_audios(df, limit=10000):
    df = df.loc[df['duration']<limit]
    return df

def dynamic_pad_batch(x):
    def not_discarded(x):
        if x is None:
            return False
        else:
            return not any([xi is None for xi in x.values()])

    def get_len(x):
        if x.ndim == 0:
            return 1
        else:
            return x.shape[0]

    def pad_to_len(x, max_len):
        if x.ndim == 0:
            return x
        else:
            pad_spec = ((0,max_len-x.shape[0]),) + ((0,0),)*(x.ndim - 1)
            return np.pad(x,pad_spec)

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            if x.dtype in [np.float64, np.float32, np.float16, 
                        np.complex64, np.complex128, 
                        np.int64, np.int32, np.int16, np.int8,
                        np.uint8, np.bool]:

                return torch.from_numpy(x)
            else:
                return x
            
    x_ = x
    x = [xi for xi in x if not_discarded(xi)]

    batch = {k: [np.array(xi[k]) for xi in x] for k in x[0]}
    batch_lens = {k: [get_len(x) for x in batch[k]] for k in batch.keys()}
    batch_max_lens = {k: max(v) for k,v in batch_lens.items()}
    batch = {k: np.stack([pad_to_len(x, batch_max_lens[k]) for x in batch[k]]) for k in batch.keys()}
    batch_lens = {k+'_lens': np.array(v) for k,v in batch_lens.items()}
    batch.update(batch_lens)
    batch = {k: to_torch(v) for k,v in batch.items()}

    return batch

def compensate_lengths(df, chunk_length=None):
    if chunk_length is not None:
        map_idx = []
        for i, (idx, row) in enumerate(df.iterrows()):
            map_idx.extend([i]*int(max(1,row['duration']//chunk_length)))
        return map_idx
    else:
        return list(range(len(df)))

class DictDataset(Dataset):
    def __init__(self, metadata, state, out_cols, preprocessors=None, index_mapper=None, state_keys=None):
        self._metadata = metadata
        self._out_cols = out_cols
        self._state = {}
        self._state['metadata'] = metadata
        if 'classID' in state['dataset_metadata'].columns:
            if isinstance(state['dataset_metadata'].iloc[0]['classID'], np.ndarray):
                self._state['num_classes'] = len(state['dataset_metadata'].iloc[0]['classID'])
            else:
                self._state['num_classes'] = state['dataset_metadata']['classID'].max() + 1
        if state_keys is not None:
            for k in state_keys:
                if k in state:
                    self._state[k] = state[k]
        self._preprocessors = preprocessors
        if index_mapper is not None:
            self._idx_map = index_mapper(self._metadata)
        else:
            self._idx_map = list(range(len(self._metadata)))

    def __getitem__(self, idx):
        row = copy.deepcopy(self._metadata.iloc[self._idx_map[idx]])
        for p in self._preprocessors:
            row, self._state = p(row, self._state)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._idx_map)

def read_selflearning_dataset(dataset_path):
    df = pd.read_csv(Path(dataset_path, 'metadata_selftrain_dataset.csv'), names=['start','stop','filename'])
    df = df.reset_index()
    df = df.rename({'index':'filename_audio','filename':'filename_targets'},axis=1)
    return df