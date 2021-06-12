DATASET=x_dataset
CKPT_DIR=/path/to/ckpt_dir/
CKPT_FILE=model.pt
TEST_FILE=/path/to/x_dataset/raw/test.source
OUTPUT_FILE=/path/to/x_dataset/train_split_output/test.hypo
INF_RETAIN_DROPOUT=False
INFERENCE_PATAMS=4,2.0,140,55,3
BATCH_SIZE=16
LOAD_CKPT_DATA=/path/to/cnndm_dataset/bin/
ATTN_TEMP=128

python3 pred.py --ckpt_dir ${CKPT_DIR} --ckpt_file ${CKPT_FILE} \
	--test_file ${TEST_FILE} --output_file ${OUTPUT_FILE} \
	--retain_dropout ${INF_RETAIN_DROPOUT} --inference_params ${INFERENCE_PARAMS} \
	--load_ckpt_data ${LOAD_CKPT_DATA} --batch_size ${BATCH_SIZE} \
	--encoder_attn_temp ${ATTN_TEMP} --decoder_attn_temp ${ATTN_TEMP} \
	--cross_attn_temp ${ATTN_TEMP}

# Evaluation
if [[ ${DATASET} != "nyt" ]]; then
	export CLASSPATH=/path/to/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
	export LC_ALL=C.UTF-8
	reference=/path/to/x_dataset/test.target
	cat $reference | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >${OUTPUT_FILE}.target
	cat ${OUTPUT_FILE} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >${OUTPUT_FILE}.tokenized
	files2rouge ${OUTPUT_FILE}.tokenized ${OUTPUT_FILE}.target
else
	export CLASSPATH=/path/to/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
	export LC_ALL=C.UTF-8
	reference=/path/to/x_dataset/test.target
	cat $reference | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >${OUTPUT_FILE}.target
	cat ${OUTPUT_FILE} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >${OUTPUT_FILE}.tokenized
	python truncate_len_by_gold.py --input=${OUTPUT_FILE}.tokenized \
		--ref=${OUTPUT_FILE}.target --output=${OUTPUT_FILE}.tokenized.truncated
	files2rouge ${OUTPUT_FILE}.tokenized.truncated ${OUTPUT_FILE}.target
fi
