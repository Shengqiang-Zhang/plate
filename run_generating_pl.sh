# Split the train.source to run prediction simultaneously
split -l 10000 /path/to/x_dataset/train.source -d -a 3 /path/to/x_dataset/train_split/train.source

CKPT_DIR=/path/to/ckpt_dir/
CKPT_FILE=model.pt
TEST_FILE=/path/to/x_dataset/raw/test.source
OUTPUT_DIR=/path/to/x_dataset/train_split_output/test.source.pred
INF_RETAIN_DROPOUT=False
INFERENCE_PATAMS=4,2.0,140,55,3
BATCH_SIZE=16
LOAD_CKPT_DATA=/path/to/cnndm_dataset/bin/
ATTN_TEMP=128

for file in /path/to/x_dataset/train_split/*; do
	TEST_FILE=${file##*/}
    OUTPUT_FILE=${OUTPUT_DIR}/${TEST_FILE}.pred
	python3 pred.py --ckpt_dir ${CKPT_DIR} --ckpt_file ${CKPT_FILE} \
            --test_file ${TEST_FILE} --output_file ${OUTPUT_FILE} \
            --retain_dropout ${INF_RETAIN_DROPOUT} --inference_params ${INFERENCE_PARAMS} \
            --load_ckpt_data ${LOAD_CKPT_DATA} --batch_size ${BATCH_SIZE} \
            --encoder_attn_temp ${ATTN_TEMP} --decoder_attn_temp ${ATTN_TEMP} \
            --cross_attn_temp ${ATTN_TEMP}
done

# Merge all the split
cat ${OUTPUT_DIR}/* > /path/to/x_dataset_attn_${ATTN_TEMP}/raw/train.target
