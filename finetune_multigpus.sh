DATA_PATH=$1
BART_PATH=$2

TOTAL_NUM_UPDATES=$3
MAX_EPOCH=$4
MIN_LR=$5
WARMUP_UPDATES=$6
LR=$7
MAX_TOKENS=$8
# MAX_SENTENCES=$8

UPDATE_FREQ=$9
REQUIRED_BATCH_SIZE_MULTIPLE=${10}

GPU=${11}  # The GPUs used, e.g., 0,1,2,3,4,5,6,7 for 8 GPUs
TRAIN_DIR=${12}  # The directory for the binarized dataset
DECODER_LAYERS=${13}  # use 0,1,2,3,4,5 for 6 decoding layers
DATASET=${14}  # cnndm_fullstops, xsum, nyt
TRAIN_FROM_SCRATCH=${15}  # False for fine-tuning with BART, True for training with Transformer-base

export PATH=~/.local/bin/:$PATH
if command -v fairseq >/dev/null 2>&1; then
    :
else
    # echo -e "y" | pip uninstall fairseq
    pip install fairseq==0.9.0 --user
    # pip install fairseq --user
fi
# -----install rouge-----
if command -v rouge >/dev/null 2>&1; then
    :
else
    pip install rouge --user
fi
# -----install file2rouge-----
if command -v files2rouge >/dev/null 2>&1; then
    :
else
    pip install -U git+https://github.com/pltrdy/pyrouge --user
    git clone https://github.com/pltrdy/files2rouge.git
    cd files2rouge
    echo -e "\n" | python setup_rouge.py
    python setup.py install --user
    cd ../
fi
# -----install java-----
if command -v java >/dev/null 2>&1; then
    :
else
    sudo apt update -y
    sudo apt -y --force-yes install default-jre
    sudo apt-get install -f -y
    sudo apt -y --force-yes install default-jre
    java -version
fi
# -----stanford-corenlp-----
export CLASSPATH=../../package/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
export LC_ALL=C.UTF-8
nvidia-smi

if [[ $DATASET == "xsum" ]] || [[ $DATASET == "nyt" ]]; then
    source=article
    target=summary
else
    source=source
    target=target
fi

echo "Train with ${DATA_PATH}/${TRAIN_DIR}"

if [[ $TRAIN_FROM_SCRATCH != True ]]; then
    echo "Training from scratch: False"
    CUDA_VISIBLE_DEVICES=$GPU fairseq-train ${DATA_PATH}/${TRAIN_DIR} \
        --restore-file $BART_PATH \
        --max-tokens $MAX_TOKENS \
        --max-epoch $MAX_EPOCH \
        --min-lr $MIN_LR \
        --task translation \
        --source-lang $source --target-lang $target \
        --truncate-source \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --required-batch-size-multiple $REQUIRED_BATCH_SIZE_MULTIPLE \
        --reset-optimizer --reset-dataloader --reset-meters \
        --arch bart_large \
        --encoder-layers-to-keep 0,1,2,3,4,5,6,7,8,9,10,11 \
        --decoder-layers-to-keep $DECODER_LAYERS \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --clip-norm 0.1 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters
elif [[ $TRAIN_FROM_SCRATCH == True ]]; then
    echo "Training from scratch: True"
    CUDA_VISIBLE_DEVICES=$GPU fairseq-train ${DATA_PATH}/${TRAIN_DIR} \
        --max-tokens $MAX_TOKENS \
        --max-epoch $MAX_EPOCH \
        --min-lr $MIN_LR \
        --task translation \
        --source-lang $source --target-lang $target \
        --truncate-source \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple $REQUIRED_BATCH_SIZE_MULTIPLE \
        --arch transformer \
        --encoder-layers-to-keep 0,1,2,3,4,5 \
        --decoder-layers-to-keep 0,1,2,3,4,5 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.2 --attention-dropout 0.1 \
        --weight-decay 0.0001 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --lr $LR --warmup-updates 4000 \
        --warmup-init-lr 1e-07 \
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters
else
    echo "param TRAIN_FROM_SCRATCH setting error"
fi

# --criterion label_smoothed_cross_entropy \
        # --reset-optimizer --reset-dataloader --reset-meters \

# CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_PATH"cnn_dm-bin/" \
#     --restore-file checkpoints/checkpoint_best.pt \
#     --max-tokens $MAX_TOKENS \
#     --max-epoch $MAX_EPOCH \
#     --min-lr $MIN_LR \
#     --task translation \
#     --source-lang article --target-lang summary \
#     --truncate-source \
#     --layernorm-embedding \
#     --share-all-embeddings \
#     --share-decoder-input-output-embed \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --required-batch-size-multiple $REQUIRED_BATCH_SIZE_MULTIPLE \
#     --arch bart_large \
#     --encoder-layers-to-keep 0,1,2,3,4,5,6,7,8,9,10,11 \
#     --decoder-layers-to-keep $DECODER_LAYERS \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --clip-norm 0.1 \
#     --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
#     --fp16 --update-freq $UPDATE_FREQ \
#     --skip-invalid-size-inputs-valid-test \
#     --find-unused-parameters;

# -----Evaluation-----
if [[ $DATASET == "cnndm_fullstops" ]]; then
    reference=${DATA_PATH}/raw/test.target
elif [[ $DATASET == "xsum" ]]; then
    reference=${DATA_PATH}/raw/test.summary
elif [[ $DATASET == "cnndm_attention_scale" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "cnndm_fullstops_attention_scale" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "cnndm_unilm_pl" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "xsum_attention_scale" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "gigaword" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "gigaword_attention_scale" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "nyt" ]] || [[ $DATASET == "nyt_attention_scale" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
elif [[ $DATASET == "pubmed" ]] || [[ $DATASET == "pubmed_attention_scale" ]]; then
    reference=$(dirname ${DATA_PATH}/${TRAIN_DIR})/raw/test.target
fi

python inference_multi_gpu.py --dataset ${DATASET} --ckpt_dir checkpoints/ \
    --train_dir "${TRAIN_DIR}" --decode_script inference.py

if [[ $DATASET == "nyt" ]] || [[ $DATASET == "nyt_attention_scale" ]]; then
    echo "Evaluation with limited-length recall based rouge"
    for ckpt in $(find checkpoints/ -type f -name \*.pt); do
        ckpt_file=${ckpt##*/}
        echo "Evaluation ckpt_file: ${ckpt_file}"
        # Tokenize hypothesis and target files.
        cat checkpoints/${ckpt_file}.test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/${ckpt_file}.test.hypo.tokenized
        if [ ! -e checkpoints/test.hypo.target ]; then
            cat $reference | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/test.hypo.target
        fi
        python truncate_len_by_gold.py --input=checkpoints/"${ckpt_file}".test.hypo.tokenized \
            --ref=checkpoints/test.hypo.target --output=checkpoints/"${ckpt_file}".test.hypo.tokenized.truncated
        files2rouge checkpoints/"${ckpt_file}".test.hypo.tokenized.truncated checkpoints/test.hypo.target
        if [ $? -ne 0 ]; then
            echo "files2rouge not executed successfully, reinstall files2rouge"
            pip install -U git+https://github.com/pltrdy/pyrouge --user
            git clone https://github.com/pltrdy/files2rouge.git
            cd files2rouge
            echo -e "\n" | python setup_rouge.py
            python setup.py install --user
            cd ../
            files2rouge checkpoints/"${ckpt_file}".test.hypo.tokenized.truncated checkpoints/test.hypo.target
        fi
    done
else
    for ckpt in $(find checkpoints/ -type f -name \*.pt); do
        ckpt_file=${ckpt##*/}
        echo "Evaluation ckpt_file: ${ckpt_file}"
        # Tokenize hypothesis and target files.
        cat checkpoints/${ckpt_file}.test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/${ckpt_file}.test.hypo.tokenized
        if [ ! -e checkpoints/test.hypo.target ]; then
            cat $reference | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/test.hypo.target
        fi
        files2rouge checkpoints/${ckpt_file}.test.hypo.tokenized checkpoints/test.hypo.target
        if [ $? -ne 0 ]; then
            echo "files2rouge not executed successfully, reinstall files2rouge"
            pip install -U git+https://github.com/pltrdy/pyrouge --user
            git clone https://github.com/pltrdy/files2rouge.git
            cd files2rouge
            echo -e "\n" | python setup_rouge.py
            python setup.py install --user
            cd ../
            files2rouge checkpoints/${ckpt_file}.test.hypo.tokenized checkpoints/test.hypo.target
        fi
    done
fi
