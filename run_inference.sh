python3 inference.py \
  --ckpt "checkpoints/rnn/best.pt" \
  --data_dir "data/AP0004_Midterm&Final_translation_dataset_zh_en" \
  --test_file "test.jsonl" \
  --device auto \
  --batch_size 64 \
  --eval_samples 0

