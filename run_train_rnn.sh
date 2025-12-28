# python3 train_rnn.py \
#   --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
#   --train_file train_10k.jsonl \
#   --valid_file valid.jsonl \
#   --batch_size 256 \
#   --epochs 100 \
#   --wandb

# additive
# python3 train_rnn.py \
#   --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
#   --train_file train_10k.jsonl \
#   --valid_file valid.jsonl \
#   --batch_size 256 \
#   --epochs 50 \
#   --seed 42 \
#   --attention_type additive \
#   --wandb \
#   --wandb_name "rnn_attn-additive_bs256_ep100_seed42"

# # dot
# python3 train_rnn.py \
#   --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
#   --train_file train_10k.jsonl \
#   --valid_file valid.jsonl \
#   --batch_size 256 \
#   --epochs 50 \
#   --seed 42 \
#   --attention_type dot \
#   --wandb \
#   --wandb_name "rnn_attn-dot_bs256_ep100_seed42"

# general
python3 train_rnn.py \
  --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
  --train_file train_10k.jsonl \
  --valid_file valid.jsonl \
  --batch_size 256 \
  --epochs 50 \
  --seed 42 \
  --attention_type general \
  --wandb \
  --wandb_name "rnn_attn-general_bs256_ep100_seed42"

