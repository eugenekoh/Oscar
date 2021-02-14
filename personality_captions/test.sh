#!/bin/bash
python personality_captions/run_pic.py \
	--do_test \
	--do_eval \
	--per_gpu_eval_batch_size 2 \
	--data_dir datasets/personality_captions \
	--num_beams 5 \
	--max_gen_length 20 \
	--eval_model_dir experiments/inject/checkpoint-12-9000 \
	--num_workers 1 
