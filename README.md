This code repository is for the paper [_Attention Temperature Matters in Abstractive Summarization Distillation_](https://arxiv.org/abs/2106.03441).

# Environment Setup
## Requirements and Installation
* python verison >= 3.6
* pytorch version >= 1.5.0
* fairseq version == 0.9.0
* [files2rouge](https://github.com/pltrdy/files2rouge)
* java version >= 11.0.11


# Data preprocessing
We follow the preprocesssing process(bpe process and binarize process) as shown in the [instruction in the fairseq toolkit](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/bart/README.cnn.md). 


# Training and Inference

## Fine-tuning the teacher model
We didn't make any changes to the code of the training process. So you can use the `fairseq-train` command as the [instruction in fairseq toolkit](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/bart/README.cnn.md) to fine-tune the teacher model.

## Generating pseudo labels
To change the attention temperature to a higher value, we should change the following line (line 13) in the file `fairseq_src/fairseq/modules/multihead_attention.py`
```python
attn_scale_ratio = 1.0  # You can change the value 1.0 to other values like 1.5 and 2.0
```
If we want to use the random attention temperature sampled from a uniform distribution, we should change the following line (line 14) in the same file above.
```python
attn_scale_ratio = -1 * torch.rand(1).cuda() + 2.0
```
We then generate pseudo labels with the teacher that has higher attention temperature.
To accelerate the generating process, we split the source document in train data to multiple pieces and use the teacher to inference on each piece simultaneously.
```bash
split -l 10000 /path/to/cnndm_dataset/train.source -d -a 3 /path/to/cnndm_dataset_pl/train.source
```
We provide the `pred.py` to predict on the input file. You can use a separate virtual environment in which the fairseq is not installed to ensure that the dependencies called from the `pred.py` are all under the source directory `fairseq_src`. 
```python
DATASET=cnndm_fullstops
CKPT_PATH=/path/to/ckpt_dir/
TEST_FILE=/path/to/cnndm_dataset/test.source
OUTPUT_FILE=/path/to/output/test.hypo
RETAIN_DROPOUT=False
INFERENCE_PATAMS=4,2.0,140,55,3
LOAD_CKPT_DATA=/path/to/cnndm_dataset/bin/
python3 pred.py --dataset ${DATASET} --ckpt_path ${CKPT_PATH} \
            --test_file ${TEST_FILE} --output_file ${OUTPUT_FILE} \
            --retain_dropout ${INF_RETAIN_DROPOUT} --inference_params ${INFERENCE_PARAMS} \
            --load_ckpt_data ${LOAD_CKPT_DATA}
```
After all predictions finish, we should merge all the predictions into the whole target file of the source document
```bash
cat /path/to/prediction_output/* > /path/to/pseudo_labels/raw/train.target
```

## Layer Reordering
To choose some specified layers like the 1st, 6th, 11th layer, we first reorder the specified layer weights to the first several layers with the script `layer_reorder.py`. For example, we want to use the 1st, 6th, 11th layer, we reorder these three layers' weights to the first three layers, i.e., 0th, 1st, 2nd layer.
```bash
python layer_reorder.py --orig_ckpt /path/to/Model/bart.large/model.pt --out_ckpt /path/to/Model/bart.large/model_0-6-11.pt --encoder-layers-to-keep 0,1,2,3,4,5,6,7,8,9,10,11 --decoder-layers-to-keep 0,6,11
```
Then use the path of the output model model_0-6-11.pt as the `$BART_PATH` to initialize the fine-tuning.

## Fine-tuning the student model 
Before fine-tuning, it's necessary to do the data pre-processing as the same process of training(i.e., BPE process and binarize process).
Then, we recommend to use the script `finetune_multigpus.sh` to fine-tune the student model on summarization datasets.
```bash
# Fine-tuning on cnndm dataset with BART 12-12 on 8 GPUs
bash finetune_multigpus.sh /path/to/cnndm_dataset/ /path/to/pretrained_bart/model.pt 20000 6 -1 500 9e-5 2048 4 1 0,1,2,3,4,5,6,7 /path/to/cnndm_dataset/bin/ 0,1,2,3,4,5,6,7,8,9,10,11 cnndm_fullstops False
# Fine-tuning on xsum dataset with BART 12-3 on 8 GPUs
bash finetune_multigpus.sh /path/to/xsum_dataset/ /path/to/pretrained_bart/model.pt 20000 13 -1 500 9e-5 2048 4 1 0,1,2,3,4,5,6,7 /path/to/xsum_dataset/bin/ 0,1,2 xsum False
# Training on nyt dataset with Transformer base 6-6 on 8 GPUs
bash finetune_multigpus.sh /path/to/nyt_dataset/ /path/to/pretrained_bart/model.pt 20000 100 -1 500 5e-4 2048 8 1 0,1,2,3,4,5,6,7 /path/to/nyt_dataset/bin/ 0,1,2,3,4,5 nyt True
```


# Evaluation
We tokenize the prediction with the standford-corenlp toolkit, then use the files2rouge to evaluate.
For the CNNDM and XSum dataset, we follow the standard full-length F1 based ROUGE.
```bash
export CLASSPATH=/path/to/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
export LC_ALL=C.UTF-8
reference=/path/to/dataset/test.target
cat $reference | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/test.hypo.target
cat checkpoints/test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/test.hypo.tokenized
files2rouge checkpoints/test.hypo.tokenized checkpoints/test.hypo.target
```
For the NYT dataset, we use the limited-length recall based ROUGE.
```bash
export CLASSPATH=/path/to/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
export LC_ALL=C.UTF-8
reference=/path/to/dataset/test.target
cat $reference | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/test.hypo.target
cat checkpoints/test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines >checkpoints/test.hypo.tokenized
python truncate_len_by_gold.py --input=checkpoints/test.hypo.tokenized \
            --ref=checkpoints/test.hypo.target --output=checkpoints/test.hypo.tokenized.truncated
files2rouge checkpoints/test.hypo.tokenized.truncated checkpoints/test.hypo.target
```
