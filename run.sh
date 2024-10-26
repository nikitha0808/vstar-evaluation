CUDA_VISIBLE_DEVICES=0 python vstar_eval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder <data_path>/vstar_bench/ \
    --answers-file <out_dir>/vstar.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
