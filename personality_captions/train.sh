#!/bin/bash
python3 personality_captions/run_pic.py \
	--model_name_or_path models/base-vg-labels/ep_107_1192087 \
	--do_train \
	--do_lower_case \
	--evaluate_during_training \
	--add_od_labels \
	--learning_rate 0.00003 \
	--per_gpu_train_batch_size 1 \
	--per_gpu_eval_batch_size 1 \
	--num_train_epochs 1 \
	--save_steps 5000 \
	--output_dir output \
	--data_dir datasets/personality_captions \
	--train_yaml train.yaml \
	--val_yaml val.yaml \
	--num_workers 4 \
	--tensorboard_log_dir runs/pic \
	--global_step_offset 80

