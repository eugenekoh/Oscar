python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --data_dir datasets/personality_captions \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir models/checkpoint-29-66420