cd ../
python3 -W ignore main.py \
--model_iters 12 \
--compile_model \
--crop_len 151 \
--configs configs/spike2flow_pp.yml \
--logs_file_name spike2flow_pp \
--eval_vis eval_vis \
--vis_path vis \
--model_name spike2flow_pp \
--pretrained ckpt/spike2flow_pp.pth \
--eval