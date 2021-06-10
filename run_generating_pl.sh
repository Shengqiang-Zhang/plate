# Split the train.source to run prediction simultaneously
split -l 10000 /path/to/x_dataset/train.source -d -a 3 /path/to/x_dataset/train_split/train.source

DATASET=cnndm_fullstops
CKPT_PATH=/path/to/ckpt_dir/
OUTPUT_DIR=/path/to/x_dataset/train_split_output/
RETAIN_DROPOUT=False
INFERENCE_PATAMS=4,2.0,140,55,3
LOAD_CKPT_DATA=/path/to/cnndm_dataset/bin/
ATTN_TEMP=128

for file in /path/to/x_dataset/train_split/*; do
	TEST_FILE=${file##*/}
    OUTPUT_FILE=${OUTPUT_DIR}/${TEST_FILE}.pred
	python3 pred.py --dataset ${DATASET} --ckpt_path ${CKPT_PATH} \
            --test_file ${TEST_FILE} --output_file ${OUTPUT_FILE} \
            --retain_dropout ${INF_RETAIN_DROPOUT} --inference_params ${INFERENCE_PARAMS} \
            --load_ckpt_data ${LOAD_CKPT_DATA} \
            --encoder_attn_temp ${ATTN_TEMP} --decoder_attn_temp ${ATTN_TEMP} \
            --cross_attn_temp ${ATTN_TEMP}
done

# Merge all the split
cat ${OUTPUT_DIR}/* > /path/to/x_dataset_attn_${ATTN_TEMP}/raw/train.target
