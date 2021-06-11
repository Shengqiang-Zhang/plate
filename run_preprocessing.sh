raw_data_dir=/path/to/x_dataset/raw/
bin_data_dir=/path/to/x_dataset/bin/

# BPE Preprocess
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "${raw_data_dir}/$SPLIT.$LANG" \
    --outputs "${raw_data_dir}/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

# Binarize dataset
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${raw_data_dir}/train.bpe" \
  --validpref "${raw_data_dir}/val.bpe" \
  --destdir "${bin_data_dir}" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
