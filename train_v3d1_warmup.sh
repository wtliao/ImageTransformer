GPU_id=5
module="3d1" #3_no_c, 3_no_p, 3d1, 3d2, 3d3,3d1w2,3d1w4
id="h_v"$module
caption_model="aoa"$module
refine_aoa=1
aoa_num=3  # 6 is set in AOA, -1 , -2, and -3 are only for ablation study "l1" "l2" and "l3"
#word_count_threshold=4
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/tmp/train_ours/log_$id"
fi
# start_from=""
# start_from="--start_from log/tmp/train_ours/log_refine_aoa_h_v3d1_aoa3_bs16_warmup"
# echo $word_count_threshold
# echo $start_from
# CUDA_VISIBLE_DEVICES=$GPU_id 
# CUDA_VISIBLE_DEVICES=$GPU_id python train_h2.py --id $id --refine_aoa $refine_aoa --caption_model $caption_model --aoa_num $aoa_num\
# 	--caption_model $caption_model \
# 	--refine 1 \
# 	--refine_aoa $refine_aoa \
# 	--aoa_num $aoa_num\
# 	--use_ff 0 \
# 	--decoder_type AoA \
# 	--use_multi_head 2 \
# 	--num_heads 8 \
# 	--multi_head_scale 1 \
# 	--mean_feats 1 \
# 	--ctx_drop 1 \
# 	--dropout_aoa 0.3 \
# 	--label_smoothing 0.2 \
# 	--input_json data/tmp/4/cocotalk.json \
# 	--input_label_h5 data/tmp/4/cocotalk_label.h5 \
# 	--input_fc_dir  data/adaptive/cocobu_fc \
# 	--input_att_dir  data/adaptive/cocobu_att  \
# 	--input_box_dir  data/adaptive/cocobu_box \
# 	--input_flag_dir data/tmp/cocobu_flag_h_v1 \
# 	--seq_per_img 5 \
# 	--batch_size 24 \
# 	--beam_size 1 \
# 	--learning_rate 2e-4 \
# 	--num_layers 2 \
# 	--input_encoding_size 1024 \
# 	--rnn_size 1024 \
# 	--learning_rate_decay_start 0 \
# 	--scheduled_sampling_start 0 \
# 	--name_append ""\
# 	--checkpoint_path log/tmp/train_ours/log_refine_aoa_${id}_aoa${aoa_num}  \
# 	$start_from \
# 	--save_checkpoint_every 5000 \
# 	--language_eval 1 \
# 	--val_images_use -1 \
# 	--max_epochs 50 \
# 	--scheduled_sampling_increase_every 5 \
# 	--scheduled_sampling_max_prob 0.5 \
# 	--learning_rate_decay_every 3 \
# 	--use_warmup 1 \
# 	--refine_lr_decay 0.9

python train_h2.py --id $id --refine_aoa $refine_aoa --caption_model $caption_model --aoa_num $aoa_num\
	--caption_model $caption_model \
	--refine 1 \
	--refine_aoa $refine_aoa \
	--aoa_num $aoa_num\
	--use_ff 0 \
	--decoder_type AoA \
	--use_multi_head 2 \
	--num_heads 8 \
	--multi_head_scale 1 \
	--mean_feats 1 \
	--ctx_drop 1 \
	--dropout_aoa 0.3 \
	--input_json data/tmp/4/cocotalk.json \
	--input_label_h5 data/tmp/4/cocotalk_label.h5 \
	--input_fc_dir  data/adaptive/cocobu_fc \
	--input_att_dir  data/adaptive/cocobu_att  \
	--input_box_dir  data/adaptive/cocobu_box \
	--input_flag_dir data/tmp/cocobu_flag_h_v1 \
	--seq_per_img 5 \
	--batch_size 96 \
	--beam_size 1 \
	--num_layers 2 \
	--input_encoding_size 1024 \
	--rnn_size 1024 \
	--language_eval 1 \
	--val_images_use -1 \
	--save_checkpoint_every 5000 \
	--name_append "38"\
	--start_from log/tmp/train_ours/log_refine_aoa_${id}_aoa${aoa_num}_bs96_warmup_gpu8\
	--checkpoint_path log/tmp/train_ours/log_refine_aoa_${id}_aoa${aoa_num}_best38_rl \
	--learning_rate 2e-5 \
	--max_epochs 80 \
	--self_critical_after 0 \
	--learning_rate_decay_start -1 \
	--scheduled_sampling_start -1 \
	--reduce_on_plateau \
	--use_warmup 1 \
	--refine_lr_decay 0.9
	
