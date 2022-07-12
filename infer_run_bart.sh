set -v

TEST_FILE="test_no_tag"
GPU=3

MODEL_PATH=model_save/bart_base_/checkpoint-10479
CUDA_VISIBLE_DEVICES=${GPU} python run_summarization.py  \
     	--model_name_or_path  $MODEL_PATH \
       	--do_predict  \
       	--test_file data/${TEST_FILE}.json  \
       	--output_dir model_save/tmp \
	--overwrite_output_dir    \
      	--per_device_eval_batch_size=64  \
     	--predict_with_generate \
	--text_column text \
	--summary_column summary \


