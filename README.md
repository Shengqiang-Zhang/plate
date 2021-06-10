This code repository is for the paper [_Attention Temperature Matters in Abstractive Summarization Distillation_](https://arxiv.org/abs/2106.03441).

# Environment Setup
## Requirements and Installation
* python verison >= 3.6
* pytorch version >= 1.5.0
* fairseq version == 0.9.0
* [files2rouge](https://github.com/pltrdy/files2rouge)
* java version >= 11.0.11


# Data preprocessing
We follow the preprocesssing process(bpe process and binarize process) as shown in the [instruction in the fairseq toolkit](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/bart/README.cnn.md). We also provide an out-of-the-box preprocessing script `run_preprocess.sh`. You just need to change the data path `raw_data_dir` and `bin_data_dir` to your own path.


# Training and Inference

## Fine-tuning the teacher model
We didn't make any changes to the code of the training process. So you can use the `fairseq-train` command or `python train.py` command as the [instruction in fairseq toolkit](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/bart/README.cnn.md) to fine-tune the teacher model.

```bash
TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/path/to/bart/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_src/train.py /path/to/x_dataset/bin/ \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```

## Generating pseudo labels
The script `pred.py` is for predicting on single file. We can change the argument value `args.encoder_attn_temp`, `args.decoder_attn_temp`, and `args.cross_attn_temp` in `pred.py` file to change the attention temperature during the teacher's inference process.

If we want to use a temperature value sampled from a uniform distribution U[64, 128], we can set:
```python
args.encoder_attn_temp = -64 * torch.rand(1).cuda() + 128
```
We can use the same method to change the decoder attention temperature and cross attention temperature.

We then generate pseudo labels with the teacher that has higher attention temperature.
We provide the `pred.py` to predict on the input file. You can use a separate virtual environment in which the fairseq is not installed to ensure that the dependencies called from the `pred.py` are all under the source directory `fairseq_src`. 

We provide a script `run_generating_pl.sh` to generate pseudo labels.
To accelerate the generating process, we split the source document of train data to multiple pieces and use the teacher to inference on each piece simultaneously. When all the predictions finish, we merge all the predictions to the needed train.target file.

## Reordering Layers
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
