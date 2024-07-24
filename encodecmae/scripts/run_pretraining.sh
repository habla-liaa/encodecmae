#-----------------------------------------First Iteration Models-------------------------------------------------------
#EC-EC Small model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name ec-ec-small_model \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=5

#EC-EC Base model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name ec-ec-base_model \
		--experiment_name upstream_model

#EC-EC Large model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name ec-ec-large_model \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=20 DEVICE="[0,1]" MODEL_DIM=1024 TRAIN_BATCH_SIZE=64 "pl.Trainer.strategy='ddp_find_unused_parameters_true'"

#Mel256-EC Small model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-small_model \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=5

#Mel256-EC Base model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model \
		--experiment_name upstream_model

#Mel256-EC Base model (Audioset)
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-as \
		--experiment_name upstream_model

#Mel256-EC Base model (Librilight)
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-ll \
		--experiment_name upstream_model

#Mel256-EC Base model (FMA)
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/fma-large-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-fma \
		--experiment_name upstream_model

#Mel256-EC Large model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-large_model \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=20 DEVICE="[0,1]" MODEL_DIM=1024 TRAIN_BATCH_SIZE=64 "pl.Trainer.strategy='ddp_find_unused_parameters_true'"

#Mel256-EC Large model - Audioset
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-large_model-as \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=20 DEVICE="[0,1]" MODEL_DIM=1024 TRAIN_BATCH_SIZE=64 "pl.Trainer.strategy='ddp_find_unused_parameters_true'"

#-----------------------------------------------ST Models--------------------------------------------------------------

#Mel256->EC Base - NoPN ST
ginpipe configs/base/create_st_dataset.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model \
		--experiment_name self_training_dataset_no-pn \
		--mods "TEACHER_MODEL='mel256-ec-base'" \
			   "POSTNORM_LAST_ACTIVATION=False" \
			   "DEVICE='cuda:0'"

ginpipe configs/base/self-train.gin \
		configs/models/encodecmae.gin \
		configs/features/mel.gin \
		configs/features/st.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model \
		--experiment_name st_model-no-pn \
		--mods "ST_CHECKPOINT='huggingface:lpepino/encodecmae-pretrained/upstreams/mel256-ec-base.pt'" \
				"ST_DATASET_DIR='/workspace/encodecmae/encodecmae/experiments/mel256-ec-base_model/self_training_dataset_no-pn'" \
				"DEVICE=[0]"

#Mel256->EC Base - NoPN ST (AS)
ginpipe configs/base/create_st_dataset.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-as \
		--experiment_name self_training_dataset_no-pn \
		--mods "TEACHER_MODEL='mel256-ec-base-as'" \
			   "POSTNORM_LAST_ACTIVATION=False" \
			   "DEVICE='cuda:0'"

ginpipe configs/base/self-train.gin \
		configs/models/encodecmae.gin \
       configs/features/mel.gin \
       configs/features/st.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-as \
		--experiment_name st_model-no-pn \
		--mods "ST_CHECKPOINT='huggingface:lpepino/encodecmae-pretrained/upstreams/mel256-ec-base-as.pt'" \
              "ST_DATASET_DIR='/workspace/encodecmae/encodecmae/experiments/mel256-ec-base_model-as/self_training_dataset_no-pn'" \
			   "DEVICE=[0]" \
			   "pl.Trainer.check_val_every_n_epoch=None"

#Mel256->EC Base - NoPN ST (LL)
ginpipe configs/base/create_st_dataset.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-ll \
		--experiment_name self_training_dataset_no-pn \
		--mods "TEACHER_MODEL='mel256-ec-base-ll'" \
			   "POSTNORM_LAST_ACTIVATION=False" \
			   "DEVICE='cuda:1'"

ginpipe configs/base/self-train.gin \
  		configs/models/encodecmae.gin \
        configs/features/mel.gin \
        configs/features/st.gin \
  		--module_list configs/imports \
  		--project_name mel256-ec-base_model-ll \
  		--experiment_name st_model-no-pn \
  		--mods "ST_CHECKPOINT='huggingface:lpepino/encodecmae-pretrained/upstreams/mel256-ec-base-ll.pt'" \
               "ST_DATASET_DIR='/workspace/encodecmae/encodecmae/experiments/mel256-ec-base_model-ll/self_training_dataset_no-pn'" \
  			   "DEVICE=[1]" \
			   "pl.Trainer.check_val_every_n_epoch=None"

#Mel256->EC Base - NoPN ST (FMA)
ginpipe configs/base/create_st_dataset.gin \
		configs/datasets/fma-large-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-base_model-fma \
		--experiment_name self_training_dataset_no-pn \
		--mods "TEACHER_MODEL='mel256-ec-base-fma'" \
			   "POSTNORM_LAST_ACTIVATION=False" \
			   "DEVICE='cuda:1'"

ginpipe configs/base/self-train.gin \
    	configs/models/encodecmae.gin \
        configs/features/mel.gin \
        configs/features/st.gin \
    	--module_list configs/imports \
    	--project_name mel256-ec-base_model-fma \
    	--experiment_name st_model-no-pn \
    	--mods "ST_CHECKPOINT='huggingface:lpepino/encodecmae-pretrained/upstreams/mel256-ec-base-fma.pt'" \
               "ST_DATASET_DIR='/workspace/encodecmae/encodecmae/experiments/mel256-ec-base_model-fma/self_training_dataset_no-pn'" \
    		   "DEVICE=[1]" \
  			   "pl.Trainer.check_val_every_n_epoch=None"

#Mel256-EC Large - NoPN ST
ginpipe configs/base/create_st_dataset.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/mel.gin \
		--module_list configs/imports \
		--project_name mel256-ec-large_model \
		--experiment_name self_training_dataset_no-pn \
		--mods "TEACHER_MODEL='mel256-ec-base_st-nopn'" \
			   "POSTNORM_LAST_ACTIVATION=False" \
			   "DEVICE='cuda:0'"

ginpipe configs/base/self-train.gin \
   	    configs/models/encodecmae.gin \
        configs/features/mel.gin \
        configs/features/st.gin \
        --module_list configs/imports \
   	    --project_name mel256-ec-large_model \
   	    --experiment_name st_model-no-pn \
   	    --mods "ST_CHECKPOINT='huggingface:lpepino/encodecmae-pretrained/upstreams/mel256-ec-large.pt'" \
               "ST_DATASET_DIR='/workspace/encodecmae/encodecmae/experiments/mel256-ec-large_model/self_training_dataset_no-pn'" \
   			   "DEVICE=[0,1]" \
 			   NUM_ENCODER_LAYERS=20 \
 			   MODEL_DIM=1024 \
 			   TRAIN_BATCH_SIZE=64 \
 			   "pl.Trainer.strategy='ddp_find_unused_parameters_true'"

#Mel256-EC Large - NoPN ST AS
ginpipe configs/base/create_st_dataset.gin \
 		configs/datasets/audioset-unbalanced-24k.gin \
 		configs/features/mel.gin \
 		--module_list configs/imports \
 		--project_name mel256-ec-large_model-as \
 		--experiment_name self_training_dataset_no-pn \
 		--mods "TEACHER_MODEL='mel256-ec-base_st-as-nopn'" \
 			   "POSTNORM_LAST_ACTIVATION=False" \
 			   "DEVICE='cuda:1'"

ginpipe configs/base/self-train.gin \
  	    configs/models/encodecmae.gin \
        configs/features/mel.gin \
        configs/features/st.gin \
        --module_list configs/imports \
  	    --project_name mel256-ec-large_model-as \
  	    --experiment_name st_model-no-pn \
  	    --mods "ST_CHECKPOINT='huggingface:lpepino/encodecmae-pretrained/upstreams/mel256-ec-large-as.pt'" \
              "ST_DATASET_DIR='/workspace/encodecmae/encodecmae/experiments/mel256-ec-large_model-as/self_training_dataset_no-pn'" \
  			   "DEVICE=[0,1]" \
			   NUM_ENCODER_LAYERS=20 \
			   MODEL_DIM=1024 \
			   TRAIN_BATCH_SIZE=64 \
			   "pl.Trainer.strategy='ddp_find_unused_parameters_true'"
