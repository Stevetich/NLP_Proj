# python3 train_transformer.py \
#   --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
#   --train_file train_10k.jsonl \
#   --valid_file valid.jsonl \
#   --batch_size 128 \
#   --epochs 100 \
#   --dim 256 --num_layers 4 --num_heads 4 --ff_dim 1024 \
#   --dropout 0.2 \
#   --warmup_steps 400 \
#   --weight_decay 0.01 \
#   --label_smoothing 0.1 \
#   --eval_samples 0 \
#   --wandb \


# sinusoidal
# python3 train_transformer.py \
#   --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
#   --train_file train_10k.jsonl \
#   --valid_file valid.jsonl \
#   --batch_size 256 \
#   --epochs 100 \
#   --dim 256 --num_layers 4 --num_heads 4 --ff_dim 1024 \
#   --dropout 0.2 \
#   --warmup_steps 400 \
#   --weight_decay 0.01 \
#   --label_smoothing 0.1 \
#   --eval_samples 0 \
#   --pos_encoding sinusoidal \
#   --seed 42 \
#   --wandb \
#   --wandb_name "tfm_pos-sinusoidal_d256_l4_h4_ff1024_do0.2_wu400_wd0.01_ls0.1_bs256_ep100_seed42"


# learned
# python3 train_transformer.py \
#   --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
#   --train_file train_10k.jsonl \
#   --valid_file valid.jsonl \
#   --batch_size 256 \
#   --epochs 100 \
#   --dim 256 --num_layers 4 --num_heads 4 --ff_dim 1024 \
#   --dropout 0.2 \
#   --warmup_steps 400 \
#   --weight_decay 0.01 \
#   --label_smoothing 0.1 \
#   --eval_samples 0 \
#   --pos_encoding learned \
#   --seed 42 \
#   --wandb \
#   --wandb_name "tfm_pos-learned_d256_l4_h4_ff1024_do0.2_wu400_wd0.01_ls0.1_bs256_ep30_seed42"

# relative
python3 train_transformer.py \
  --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
  --train_file train_10k.jsonl \
  --valid_file valid.jsonl \
  --batch_size 256 \
  --epochs 100 \
  --dim 256 --num_layers 4 --num_heads 4 --ff_dim 1024 \
  --dropout 0.2 \
  --warmup_steps 400 \
  --weight_decay 0.01 \
  --label_smoothing 0.1 \
  --eval_samples 0 \
  --pos_encoding relative \
  --seed 42 \
  --wandb \
  --wandb_name "tfm_pos-relative_d256_l4_h4_ff1024_do0.2_wu400_wd0.01_ls0.1_bs256_ep100_seed42"
