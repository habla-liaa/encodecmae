import torch
from pathlib import Path
from loguru import logger
import torchinfo
import inspect
from tqdm import tqdm
import joblib
import numpy as np
from encodecmae.hub import load_model as load_encodecmae_model
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits

def load_model(state, model):
    model = load_encodecmae_model(model)
    state['model'] = model
    return state

def create_st_dataset(state, layer=-1, kmeans_samples=10000, device='cuda:0', postnorm_last_activation=True,kmeans_jobs=8):
    from sklearn.cluster import KMeans

    model = state['model'].to(device)
    model.visible_encoder.compile=False
    #First learn kmeans:
    if ('tokenizer' in state):
        cluster_model = state['tokenizer']
    elif Path(state['output_dir'],'tokenizer.pkl').exists():
        cluster_model = joblib.load(Path(state['output_dir'],'tokenizer.pkl'))
    else:
        kmeans_sample_idxs = np.random.choice(np.arange(0,len(state['datasets']['train'])),size=kmeans_samples,replace=False)
        kmeans_dataset = []
        for i in tqdm(kmeans_sample_idxs):
            x = state['datasets']['train'][i]
            x = state['dataloaders']['train'].collate_fn([x])
            x = {k: v.to(model.device) for k,v in x.items()}
            out = model.extract_activations(x, postnorm_last_activation=postnorm_last_activation)
            kmeans_dataset.append(out['visible_embeddings_activations'][layer].cpu().numpy())
        kmeans_dataset = np.concatenate(kmeans_dataset,axis=1)[0]
        n_tokens = model.head.num_tokens
        with threadpool_limits(user_api="blas", limits=kmeans_jobs):
            cluster_model = KMeans(n_clusters=n_tokens)
            kmeans_sample_idxs = np.random.choice(np.arange(0,kmeans_dataset.shape[0]),size=kmeans_samples,replace=False)
            kmeans_dataset = kmeans_dataset[kmeans_sample_idxs]
            cluster_model.fit(kmeans_dataset)
        state['tokenizer']=cluster_model
        joblib.dump(cluster_model, Path(state['output_dir'],'tokenizer.pkl'))
    state['datasets']['train']._out_cols+=['start','stop','filename']
    state['dataloaders']['train'] = torch.utils.data.DataLoader(state['dataloaders']['train'].dataset, shuffle=False, num_workers=4, collate_fn=state['dataloaders']['train'].collate_fn, batch_size=state['dataloaders']['train'].batch_size)
    #Find last saved index before starting this loop:
    data_out_path = Path(state['output_dir'],'self_training_dataset')
    if not data_out_path.exists():
        data_out_path.mkdir(parents=True)
    for batch_idx, x in enumerate(tqdm(state['dataloaders']['train'])):
        #out = model.extract_activations({'wav': x['wav'].to(device=model.device, dtype=model.dtype),
        #                                 'wav_lens': x['wav_lens'].to(device=model.device)}, postnorm_last_activation=postnorm_last_activation)
        x = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in x.items()}
        out = model.extract_activations(x, postnorm_last_activation=postnorm_last_activation)
        out = out['visible_embeddings_activations'][layer].cpu().numpy()
        for i in range(out.shape[0]):
            indexs = cluster_model.predict(out[i])
            filename = '{}_{}_{}.npy'.format(Path(x['filename'][i]).stem,x['start'][i],x['stop'][i])
            file_out = Path(state['output_dir'],'self_training_dataset',filename[:3],filename)
            file_out.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.save(file_out,indexs)
            except:
                print('Failed {}'.format(file_out))
            with open(Path(state['output_dir'],'metadata_selftrain_dataset.csv'),'a') as f:
                f.write('{},{},{},{}\n'.format(str(x['filename'][i]),x['start'][i],x['stop'][i],str(file_out.resolve())))
    state['model'] = state['model'].to('cpu')
    return state

def fit_model(state, trainer_cls=None, model_cls=None, from_checkpoint=None, 
              cpu_threads=8,dataloaders_key='dataloaders',
              checkpoint_folder='checkpoints',
              key_out='model',
              cache_model=True,
              keep_optimizer_state=False,
              non_strict_checkpoint=True):

    if from_checkpoint.startswith('huggingface:'):
        hf_path = from_checkpoint.split('huggingface:')[-1]
        hf_repo = '/'.join(hf_path.split('/')[:2])
        path_in_repo = '/'.join(hf_path.split('/')[2:])
        from_checkpoint = hf_hub_download(repo_id=hf_repo, filename=path_in_repo)

    if not ((key_out in state) and (cache_model)):
        torch.set_num_threads(cpu_threads)
        torch.set_float32_matmul_precision('medium')
        kwargs = {}
        if 'state' in inspect.signature(model_cls.__init__).parameters.keys():
            kwargs['state'] = state
            
        if model_cls is None:
            if Path(from_checkpoint).stem == 'state':
                model = joblib.load(from_checkpoint)['model']
                from_checkpoint=None
        else:
            model = model_cls(**kwargs)
        trainer = trainer_cls()
        trainer.checkpoint_callback.dirpath = trainer.checkpoint_callback.dirpath + '/{}'.format(checkpoint_folder)
        base_dir = trainer.checkpoint_callback.dirpath

        #Find last checkpoint
        if from_checkpoint == 'last':
            ckpts = list(Path(base_dir).glob('*.ckpt'))
            if 'last' in [x.stem for x in ckpts]:
                from_checkpoint=Path(base_dir, 'last.ckpt')
            else:
                ckpt_epoch = [int(c.stem.split('epoch=')[-1].split('-')[0]) for c in ckpts]
                if len(ckpt_epoch) > 0:
                    last_epoch = max(ckpt_epoch)
                    from_checkpoint = ckpts[ckpt_epoch.index(last_epoch)]
                else:
                    logger.info('No checkpoints found in {}. Training from scratch.'.format(base_dir))
                    from_checkpoint = None

        logger.info(torchinfo.summary(model))

        if (from_checkpoint is not None) and (non_strict_checkpoint):
            ckpt_data = torch.load(from_checkpoint)
            model_sd = model.state_dict()
            ckpt_sd = {}
            for k,v in ckpt_data['state_dict'].items():
                if (k in model_sd) and (v.shape == model_sd[k].shape):
                    ckpt_sd[k] = v
                else:
                    print("Couldn't load {} from checkpoint".format(k))

            model.load_state_dict(ckpt_sd, strict=False)
            from_checkpoint=None

        trainer.fit(model,
                    state[dataloaders_key]['train'],
                    state[dataloaders_key]['validation'],
                    ckpt_path=from_checkpoint)
        trainer.save_checkpoint(Path(base_dir,'last.ckpt'))
        state[key_out+'_checkpoint_dir'] = trainer.checkpoint_callback.dirpath
        best_model_path = model.trainer.checkpoint_callback.best_model_path
        if (best_model_path is not None) and (best_model_path != ''):
            model.load_state_dict(torch.load(best_model_path)['state_dict'])
        state[key_out] = model
    else:     
        logger.info('Model is already in state. Skipping task.')
    
    return state
