#!/bin/bash
# split train.source
DATASET=nyt

if [[ $DATASET == "cnndm_fullstops" ]]; then
    blob_train_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/cnndm_bart_fix_fullstops/cnn-dailymail/cnn_dm
    split_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/cnndm_bart_fix_fullstops/cnn-dailymail/train_split/
    TEST_DIR=../../dataset/cnndm_bart_fix_fullstops/cnn-dailymail/train_split/
    # CKPT_ID="38e294e6-5a15-463e-9938-6a373b501468"
    CKPT_ID="119abc15-3606-4782-aace-70367f801dbb" # new teacher initialized with bart.large
elif [[ $DATASET == "cnndm" ]]; then
    blob_train_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/bart_new_cnndm/cnn_dm
    split_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/bart_new_cnndm/train_split/
    TEST_DIR=../../dataset/bart_new_cnndm/train_split/
    CKPT_ID="38e294e6-5a15-463e-9938-6a373b501468"
    # CKPT_ID="119abc15-3606-4782-aace-70367f801dbb" # new teacher initialized with bart.large
elif [[ $DATASET == "xsum" ]]; then
    blob_train_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/xsum/raw
    split_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/xsum/train_split/
    TEST_DIR=../../dataset/xsum/train_split/
    # CKPT_ID="64d03b9e-ce5f-47ba-8052-911e61cf9a57"  # original
    # CKPT_ID="7dbd438c-4229-450b-b473-364c7b9f0d6a"  # dropout=0.2, 4/25, R1/R2/RL:44.605/22.171/37.034
    CKPT_ID="706cbb23-c74e-4a2a-a7ce-f8b12bd8f984"
elif [[ $DATASET == "gigaword" ]]; then
    blob_train_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/gigaword/gigaword_uncased/raw/
    split_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/gigaword/gigaword_uncased/train_split/
    TEST_DIR=../../dataset/gigaword/gigaword_uncased/train_split/
    # CKPT_ID="6173d495-73b2-4aa1-b587-8402ddbfa9dc"
    CKPT_ID="137c0adf-b74b-469e-8a76-91547e554699"
elif [[ $DATASET == "nyt" ]]; then
    blob_train_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/nyt/raw/
    split_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/nyt/train_split/
    TEST_DIR=../../dataset/nyt/train_split/
    # CKPT_ID="6173d495-73b2-4aa1-b587-8402ddbfa9dc"
    # CKPT_ID="09ad304a-00a8-475c-bcc7-a499f5e7067c"  # R2:36.085
    CKPT_ID="dce23b24-e0d4-427c-9c4d-5c425d9c10bb" # R2:36.498
elif [[ $DATASET == "pubmed" ]]; then
    blob_train_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/pubmed/raw/
    split_dir=/home/v-shezhang/blobs/readindl/shengqiang/dataset/pubmed/train_split/
    TEST_DIR=../../dataset/pubmed/train_split/
    CKPT_ID="0392b15f-9854-4981-b524-e8b4ff9853ef" # R2: 18.866 
fi

if [ ! -d $split_dir ]; then
    mkdir $split_dir
    if [[ $DATASET == "gigaword" ]];then
        split -l 100000 $blob_train_dir/train.source -d -a 3 $split_dir/train.source.
    elif [[ $DATASET == "pubmed" ]];then
        split -l 3000 $blob_train_dir/train.source -d -a 2 $split_dir/train.source.
    else
        split -l 10000 $blob_train_dir/train.source -d -a 2 $split_dir/train.source.
    fi
else
    echo "ignore split train data"
fi

attention_setting=encdec80to112
# noise_setting=attn_weight_add_gausnoise_0.001
OUTPUT_DIR="$DATASET/attention_scale_${attention_setting}"
DATA_DIR="None"  # only used when fine-tune student
# OUTPUT_DIR="$DATASET/attention_scale_${attention_setting}_dropout"
# OUTPUT_DIR="$DATASET/attention_scale_${attention_setting}_${noise_setting}"
IF_EVAL="N"
OUTPUT_NAME_WITH_CKPT=output_name_with_ckpt_no
INF_EVERY_CKPT="N"
INF_RETAIN_DROPOUT=False
# cluster=canada1GPUcl
# cluster=korea1GPUcl
cluster=japan1gpucl
# cluster=itpscusv100cl
# cluster=itpeastusv100cl2
GPU_COUNT=1
CKPT_FILE_SETTING="checkpoint_last.pt"
if [[ $DATASET == "cnndm_fullstops" ]] || [[ $DATASET == "cnndm" ]]; then
    INFERENCE_PARAMS=4,2.0,140,55,3
elif [[ $DATASET == "xsum" ]]; then
    INFERENCE_PARAMS=6,0.1,60,1,3
elif [[ $DATASET == "gigaword" ]]; then
    INFERENCE_PARAMS=6,0.1,60,1,3
elif [[ $DATASET == "nyt" ]]; then
    INFERENCE_PARAMS=4,3,350,80,3
elif [[ $DATASET == "pubmed" ]]; then
    INFERENCE_PARAMS=4,3,400,40,3
fi

for file in $split_dir/*; do
    TEST_FILE=${file##*/}
    echo "TEST FILE:$TEST_FILE"
    EXPERIMENT_TAG="teacher-pred-train,$DATASET,attn_${attention_setting},$TEST_FILE"
    sh pred_aml_submit.sh $EXPERIMENT_TAG $TEST_FILE $OUTPUT_DIR $IF_EVAL $CKPT_ID $TEST_DIR \
      $INF_EVERY_CKPT $INF_RETAIN_DROPOUT $DATASET $cluster $INFERENCE_PARAMS $OUTPUT_NAME_WITH_CKPT \
      $DATA_DIR $GPU_COUNT $CKPT_FILE_SETTING
done
