set -v

CUDA_VISIBLE_DEVICES=1 python run_summarization.py  \
	--model_name_or_path model_save/bart_base_/checkpoint-10479 \
	--do_train \
       	--num_train_epochs  16.0  \
       	--do_eval \
       	--evaluation_strategy epoch \
      	--do_predict  \
      	--train_file data/hypo_train.json   \
      	--validation_file data/hypo_dev.json  \
     	--test_file data/test_no_tag.json  \
	--output_dir model_save/mover_hypo_train_hypo-xl \
	--overwrite_output_dir    \
       	--per_device_train_batch_size=16    \
       	--per_device_eval_batch_size=16 \
    	--gradient_accumulation_steps 1 \
	--predict_with_generate \
	--text_column text \
	--save_total_limit 5 \
	--summary_column summary \
	--logging_dir logs/ \
	--save_strategy epoch \
	--learning_rate 3e-5 \
	--weight_decay 0.01 \
	--warmup_ratio 0.05
