#Small model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name small_model \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=5

#Base model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name base_model \
		--experiment_name upstream_model

#Large model
ginpipe configs/base/unsupervised_pretrain.gin \
		configs/models/encodecmae.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name large_model \
		--experiment_name upstream_model \
        --mods NUM_ENCODER_LAYERS=20 DEVICE="[0,1]" MODEL_DIM=1024 TRAIN_BATCH_SIZE=64 "pl.Trainer.strategy='ddp_find_unused_parameters_true'"

#Base model + ST
ginpipe configs/base/create_selftraining_dataset.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name base_model \
		--experiment_name self_training_dataset \
		--mods "UPSTREAM_DIR='experiments/base_model/upstream_model'" "UPSTREAM_CKPT_FILE='experiments/base_model/upstream_model/pretrain_checkpoints/last.ckpt'"

ginpipe configs/base/self-train.gin \
		configs/models/encodecmae.gin \
		configs/models/selftrain_8q_weight_by_variance.gin \
		--module_list configs/imports \
		--project_name base_model \
		--experiment_name self_training_model \
		--mods "SELFTRAIN_CHECKPOINT='/workspace/encodec_mae_refactor/experiments/base_model/upstream_model/pretrain_checkpoints/last.ckpt'" "TOTAL_PRETRAIN_STEPS=650000" "tasks.models.EncodecMAE.n_extra_targets=1" "SELFTRAINING_DATADIR='/workspace/encodec_mae_refactor/experiments/base_model/self_training_dataset'"
        
#Large model + ST

ginpipe configs/base/create_selftraining_dataset.gin \
		configs/datasets/audioset-unbalanced-24k.gin \
		configs/datasets/fma-large-24k.gin \
		configs/datasets/librilight-6k-24k.gin \
		configs/features/wav_only.gin \
		--module_list configs/imports \
		--project_name large_model \
		--experiment_name self_training_dataset \
		--mods "UPSTREAM_DIR='experiments/base_model/self_training_model'" "UPSTREAM_CKPT_FILE='experiments/base_model/self_training_model/pretrain_checkpoints/last.ckpt'"

ginpipe configs/base/self-train.gin \
		configs/models/encodecmae.gin \
		configs/models/selftrain_8q_weight_by_variance.gin \
		--module_list configs/imports \
		--project_name large_model \
		--experiment_name self_training_model \
		--mods "SELFTRAIN_CHECKPOINT='/workspace/encodec_mae_refactor/experiments/large_model/upstream_model/pretrain_checkpoints/last.ckpt'" "TOTAL_PRETRAIN_STEPS=650000" "tasks.models.EncodecMAE.n_extra_targets=1" "SELFTRAINING_DATADIR='/workspace/encodec_mae_refactor/experiments/large_model/self_training_dataset'" NUM_ENCODER_LAYERS=20 DEVICE="[0,1]" MODEL_DIM=1024 TRAIN_BATCH_SIZE=64 "pl.Trainer.strategy='ddp_find_unused_parameters_true'"
