# -*-coding:utf-8-*-
from fairseq.models.bart import BARTModel
import time
import torch
import os
import argparse
print("current path:", os.getcwd())

# aml_path = "/home/v-shezhang/blobs/readindl/shengqiang/"
# checkpoint_path = aml_path + "aml-code/9a65f8e5-de6e-41f5-bddc-3a8a525cd4a8/checkpoints/"
checkpoint_path = "./checkpoints/"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--train_dir")
parser.add_argument("--ckpt_file")
args = parser.parse_args()

if args.dataset == "cnndm_fullstops":
    test_dir = "../../dataset/cnndm_bart_fix_fullstops/cnn-dailymail/raw/"
    load_ckpt_data = "../../../dataset/cnndm_bart_fix_fullstops/cnn-dailymail/bin/"
    beam=4; lenpen=2.0; max_len_b=140; min_len=55; no_repeat_ngram_size=3
elif args.dataset == "cnndm_fullstops_attention_scale":
    test_dir = "../../dataset/cnndm_bart_fix_fullstops/cnn-dailymail/raw/"
    load_ckpt_data = "../../../dataset/cnndm_fullstops_attention_scale/" + args.train_dir
    beam=4; lenpen=2.0; max_len_b=140; min_len=55; no_repeat_ngram_size=3
elif args.dataset == "cnndm_attention_scale":
    test_dir = "../../dataset/bart_new_cnndm/cnn_dm/"
    load_ckpt_data = "../../../dataset/cnndm_attention_scale/" + args.train_dir
    beam = 4; lenpen = 2.0; max_len_b = 140; min_len = 55; no_repeat_ngram_size = 3
elif args.dataset == "cnndm_unilm_pl":
    test_dir = "../../dataset/bart_new_cnndm/cnn_dm/"
    load_ckpt_data = "../../../dataset/cnndm_unilm_pl/" + args.train_dir
    beam=4; lenpen=2.0; max_len_b=140; min_len=55; no_repeat_ngram_size=3
elif args.dataset == "xsum":
    test_dir = "../../dataset/xsum/raw/"
    load_ckpt_data = "../../../dataset/xsum/bin/"
    beam=6; lenpen=0.1; max_len_b=60; min_len=1; no_repeat_ngram_size=3
    # beam=6; lenpen=1.0; max_len_b=60; min_len=10; no_repeat_ngram_size=3
elif args.dataset == "xsum_attention_scale":
    test_dir = "../../dataset/xsum/raw/"
    load_ckpt_data = "../../../dataset/xsum_attention_scale/" + args.train_dir
    beam=6; lenpen=0.1; max_len_b=60; min_len=1; no_repeat_ngram_size=3
elif args.dataset == "gigaword":
    test_dir = "../../dataset/gigaword/gigaword_uncased/raw/"
    load_ckpt_data = "../../../dataset/gigaword/gigaword_uncased/" + args.train_dir
    beam=6; lenpen=0.7; max_len_b=40; min_len=1; no_repeat_ngram_size=3
elif args.dataset == "gigaword_attention_scale":
    test_dir = "../../dataset/gigaword/gigaword_uncased/raw/"
    load_ckpt_data = "../../../dataset/gigaword_attention_scale/" + args.train_dir
    beam=6; lenpen=0.7; max_len_b=40; min_len=1; no_repeat_ngram_size=3
elif args.dataset == "nyt":
    test_dir = "../../dataset/nyt/raw/"
    load_ckpt_data = "../../../dataset/nyt/bin/"
    beam=4; lenpen=3.0; max_len_b=350; min_len=80; no_repeat_ngram_size=3
elif args.dataset == "nyt_attention_scale":
    test_dir = "../../dataset/nyt/raw/"
    load_ckpt_data = "../../../dataset/nyt_attention_scale/" + args.train_dir
    beam=4; lenpen=3.0; max_len_b=350; min_len=80; no_repeat_ngram_size=3
elif args.dataset == "pubmed":
    test_dir = "../../dataset/pubmed/raw/"
    load_ckpt_data = "../../../dataset/pubmed/bin/"
    beam = 4; lenpen = 3.0; max_len_b = 400; min_len = 40; no_repeat_ngram_size = 3
elif args.dataset == "pubmed_attention_scale":
    test_dir = "../../dataset/pubmed/raw/"
    load_ckpt_data = "../../../dataset/pubmed_attention_scale/" + args.train_dir
    beam = 4; lenpen = 3.0; max_len_b = 400; min_len = 40; no_repeat_ngram_size = 3
else:
    print("args.dataset setting error")
    raise ValueError

print("Load pre-trained model from", args.ckpt_file, load_ckpt_data)
bart = BARTModel.from_pretrained(
    checkpoint_path,
    checkpoint_file=args.ckpt_file,
#     data_name_or_path="../../../dataset/bart_new_cnndm/bart-pseudo-label-bin"
    data_name_or_path=load_ckpt_data
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32

print("current path:", os.getcwd())
print("files under the test data directory:", os.listdir())
print("files under the checkpoint directory:", os.listdir(checkpoint_path))

elapsed_time = 0
print("Inference on the data", test_dir + "test.source")
print("Inference result will save as", checkpoint_path + args.ckpt_file + ".test.hypo")
print("Inference with parameters: beam: {}, lenpen: {}, maxlen_b: {}, minlen: {}, no_repeat_n_grams: {}"
      .format(beam, lenpen, max_len_b, min_len, no_repeat_ngram_size))
with open(test_dir + 'test.source', "r", encoding="utf-8") as source, \
        open(checkpoint_path + args.ckpt_file + '.test.hypo', 'w', encoding="utf-8") as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                time_begin = time.time()
                hypotheses_batch = bart.sample(
                    slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b,
                    min_len=min_len, no_repeat_ngram_size=no_repeat_ngram_size
                    )
                elapsed_time += time.time() - time_begin

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        time_begin = time.time()
        hypotheses_batch = bart.sample(
                slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b,
                min_len=min_len, no_repeat_ngram_size=no_repeat_ngram_size
            )
        elapsed_time += time.time() - time_begin
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
print("inference elapsed time", elapsed_time)
