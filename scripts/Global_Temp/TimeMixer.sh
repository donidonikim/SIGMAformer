export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/global_temp/ \
  --data_path None \
  --model_id global_temp_48_6_TimeMixer \
  --model TimeMixer \
  --data Global_Temp \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 6 \
  --e_layers 2 \
  --enc_in 11 \
  --d_ff 3072 \
  --des 'global_temp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --d_model 768 \
  --batch_size 1 \
  --node_num 3850 \
  --n_heads 16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/global_temp/ \
  --data_path None \
  --model_id global_temp_48_12_TimeMixer \
  --model TimeMixer \
  --data Global_Temp \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 11 \
  --d_ff 3072 \
  --des 'global_temp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --d_model 768 \
  --batch_size 1 \
  --node_num 3850 \
  --n_heads 16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/global_temp/ \
  --data_path None \
  --model_id global_temp_48_18_TimeMixer \
  --model TimeMixer \
  --data Global_Temp \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 18 \
  --e_layers 2 \
  --enc_in 11 \
  --d_ff 3072 \
  --des 'global_temp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --d_model 768 \
  --batch_size 1 \
  --node_num 3850 \
  --n_heads 16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/global_temp/ \
  --data_path None \
  --model_id global_temp_48_24_TimeMixer \
  --model TimeMixer \
  --data Global_Temp \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 11 \
  --d_ff 3072 \
  --des 'global_temp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --d_model 768 \
  --batch_size 1 \
  --node_num 3850 \
  --n_heads 16

