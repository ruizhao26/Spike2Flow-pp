cd ../
python3 -W ignore main.py \
--batch_size 6 \
--num_workers 12 \
--learning_rate 3e-4 \
--decay_interval 15 \
--decay_factor 0.7 \
--epochs 120 \
--model_iters 12 \
--no_warm 1 \
--compile_model \
--crop_len 151 \
--configs configs/spike2flow_pp.yml \
--logs_file_name spike2flow_pp \
--print_freq 100 \
--eval_vis eval_vis \
--vis_path vis \
--model_name spike2flow_pp