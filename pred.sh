#!/bin/bash
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

gpu=${11}
TEST_FILE=${12}
OUTPUT_DIR=${13}
IF_EVAL=${14}
CKPT_ID=${15}
TEST_DIR=${16}
INF_EVERY_CKPT=${17}
INF_RETAIN_DROPOUT=${18}
DATASET=${19}
INFERENCE_PARAMS=${20}
OUTPUT_NAME_WITH_CKPT=${21}
DATA_DIR=${22}
CKPT_FILE_SETTING=${23}

export PATH=~/.local/bin/:$PATH
pip install regex --user
if command -v fairseq >/dev/null 2>&1; then
    echo "fairseq run successfully"
else
    :
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
fi
java -version

# -----stanford-corenlp-----
export CLASSPATH=../../package/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
export LC_ALL=C.UTF-8

# Tokenize hypothesis and target files.
ckpt_list=$CKPT_ID
# ckpt_list="9c90bb66-87cc-4bfd-81b4-3687e1eb9714"

for ckpt_id in $(echo $ckpt_list | tr ";" "\n"); do
    for ckpt in $(find ../$ckpt_id/checkpoints/ -type f -name \*.pt); do
        ckpt_file=${ckpt##*/}
        current_dir=$(basename $PWD)
        output_file=../../dataset/new_prediction/${OUTPUT_DIR}/${current_dir}/${ckpt_id}_${ckpt_file}.${TEST_FILE}.pred
        if [[ ${INF_EVERY_CKPT} == "N" ]]; then
            if [[ ${OUTPUT_NAME_WITH_CKPT} == "output_name_with_ckpt" ]]; then
#                if [[ ${ckpt_file} != "checkpoint_last.pt" ]] && [[ ${ckpt_file} != "checkpoint_best.pt" ]]; then
#                    echo "Ignore ${ckpt_file}"
#                    continue
#                fi
                if [[ ${ckpt_file} != ${CKPT_FILE_SETTING} ]]; then
                    echo "Ignore ${ckpt_file}"
                    continue
                fi

                # if [[ $DATASET == "pubmed" ]];then
                #     if [[ ${ckpt_file} != "checkpoint10.pt" ]]; then
                #         echo "Ignore ${ckpt_file}"
                #         continue
                #     fi
                # else
                #     if [[ ${ckpt_file} != "checkpoint_last.pt" ]]; then
                #         echo "Ignore ${ckpt_file}"
                #         continue
                #     fi
                # fi

            else
                if [[ $DATASET == "nyt" ]];then
                    if [[ ${ckpt_file} != "checkpoint9.pt" ]]; then
                        echo "Ignore ${ckpt_file}"
                        continue
                    fi
                elif [[ $DATASET == "pubmed" ]];then
                    if [[ ${ckpt_file} != "checkpoint5.pt" ]]; then
                        echo "Ignore ${ckpt_file}"
                        continue
                    fi
                else
                    if [[ ${ckpt_file} != "checkpoint_last.pt" ]]; then
                        echo "Ignore ${ckpt_file}"
                        continue
                    fi

                fi
                output_file=../../dataset/new_prediction/${OUTPUT_DIR}/${TEST_FILE}.pred
            fi
        else
            :
        fi

        echo "Inference with $ckpt"

        python3 pred.py --dataset ${DATASET} --ckpt_id ${ckpt_id} --ckpt_file ${ckpt_file} \
            --test_file ${TEST_DIR}/${TEST_FILE} --output_file ${output_file} --data_dir ${DATA_DIR} \
            --retain_dropout ${INF_RETAIN_DROPOUT} --inference_params ${INFERENCE_PARAMS}

        if [[ $DATASET == "cnndm_fullstops" ]] || [[ $DATASET == "cnndm_fullstops_attention_scale" ]]; then
            reference=$DATA_PATH/cnndm_bart_fix_fullstops/cnn-dailymail/raw/test.target
        elif [[ $DATASET == "xsum" ]]; then
            reference=$DATA_PATH/xsum/raw/test.target
        elif [[ $DATASET == "gigaword" ]]; then
            reference=$DATA_PATH/gigaword/gigaword_uncased/raw/test.target
        elif [[ $DATASET == "nyt" ]]; then
            reference=$DATA_PATH/nyt/raw/test.target
        elif [[ $DATASET == "pubmed" ]]; then
            reference=$DATA_PATH/pubmed/raw/test.target
        fi

        if [[ $IF_EVAL == "Y" ]]; then
            if [[ $DATASET == "nyt" ]]; then
                echo "Evaluation with limited-length recall based rouge"
                cat "${output_file}" | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >"${output_file}".tokenized
                if [ ! -e "${output_file}".target ]; then
                    cat "${reference}" | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >"${output_file}".target
                fi

                python truncate_len_by_gold.py --input="${output_file}".tokenized \
                    --ref="${output_file}".target --output="${output_file}".tokenized.truncated
                files2rouge "${output_file}".tokenized.truncated "${output_file}".target
                if [ $? -ne 0 ]; then
                    echo "files2rouge not executed successfully, reinstall files2rouge"
                    pip install -U git+https://github.com/pltrdy/pyrouge --user
                    git clone https://github.com/pltrdy/files2rouge.git
                    cd files2rouge
                    echo -e "\n" | python setup_rouge.py
                    python setup.py install --user
                    cd ../
                    files2rouge checkpoints/"${output_file}".tokenized.truncated checkpoints/"${output_file}".target
                fi
            else
                cat ${output_file} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >${output_file}.tokenized
                if [ ! -e ${output_file}.target ]; then
                    cat ${reference} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >${output_file}.target
                fi
                files2rouge ${output_file} ${output_file}.target
                if [ $? -ne 0 ]; then
                    echo "files2rouge not executed successfully, reinstall files2rouge"
                    pip install -U git+https://github.com/pltrdy/pyrouge --user
                    git clone https://github.com/pltrdy/files2rouge.git
                    cd files2rouge
                    echo -e "\n" | python setup_rouge.py
                    python setup.py install --user
                    cd ../
                    files2rouge ${output_file} ${output_file}.target
                fi
            fi
        fi
    done
done
