python3 finetune_pretrained.py \
  --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
  --train_file train_10k.jsonl \
  --valid_file valid.jsonl \
  --test_file test.jsonl \
  --src_key zh \
  --tgt_key en \
  --model_name_or_path t5-small \
  --source_prefix "translate Chinese to English: " \
  --batch_size 16 \
  --max_src_len 200 \
  --max_tgt_len 200 \
  --gen_max_new_tokens 120 \
  --epochs 5 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --fp16 \
  --save_dir checkpoints/t5

