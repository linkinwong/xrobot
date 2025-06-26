nohup python pipeline.py --data_dir ./my_dataset --wake_word "xiaoqi" --epochs 10 --batch_size 64 --num_workers 30 --averaging_frames 4 --resplit_data > train_log_june_26_16_32 2>&1 &



nohup python pipeline.py --data_dir ./my_dataset --wake_word "xiaoqi" --epochs 10 --batch_size 64 --num_workers 30 --averaging_frames 4 --model_dim 64 --resplit_data > train_log_june_26_16_43 2>&1   &
