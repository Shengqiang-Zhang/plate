# from fairseq.models.bart import BARTModel
import sys
import os

print("current dir", os.path.dirname(__file__))
print("os.getcwd", os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "fairseq_src"))
sys.path.append(os.path.join(os.getcwd(), "fairseq_src/fairseq"))
print("sys.path", sys.path)

from fairseq_src.fairseq.models.bart import BARTModel
from pathlib import Path
import torch
import time
import argparse
import warnings

aml_path = "../../"


def pred_ckpt(
        test_file, output_file, ckpt_path, ckpt_file, batch_size, load_ckpt_data,
        encoder_attn_temp, decoder_attn_temp, cross_attn_temp,
        **eval_kwargs,
        # beam, lenpen, max_len_b, min_len, no_repeat_ngram_size
):
    assert ckpt_file.endswith(".pt"), "ckpt_file is not a checkpoint"
    print("-----Inference with", ckpt_file)
    bart = BARTModel.from_pretrained(
        ckpt_path,
        checkpoint_file=ckpt_file,
        data_name_or_path=load_ckpt_data,
        encoder_attn_temp=encoder_attn_temp,
        decoder_attn_temp=decoder_attn_temp,
        cross_attn_temp=cross_attn_temp
    )
    print("-----Inference kwargs:", eval_kwargs)

    bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    bsz = batch_size

    pred_time = 0
    with open(test_file, "r", encoding="utf-8") as source, open(output_file, 'w', encoding="utf-8") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    begin = time.time()
                    # hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=no_repeat_ngram_size)
                    hypotheses_batch = bart.sample(slines, **eval_kwargs)
                    pred_time += time.time() - begin

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            begin = time.time()
            # hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=no_repeat_ngram_size)
            hypotheses_batch = bart.sample(slines, **eval_kwargs)
            pred_time += time.time() - begin
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()

    print("pred time", pred_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--ckpt_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--retain_dropout", type=str)
    parser.add_argument("--inference_params", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--load_ckpt_data", type=str)
    parser.add_argument("--encoder_attn_temp", type=int, default=None)
    parser.add_argument("--decoder_attn_temp", type=int, default=None)
    parser.add_argument("--cross_attn_temp", type=int, default=None)
    args = parser.parse_args()

    if args.inference_params:
        params = args.inference_params.split(",")
        assert len(params) == 5, "number of inference params != 5"
        EVAL_KWARGS = {
            "beam": int(params[0]),
            "lenpen": float(params[1]),
            "max_len_b": int(params[2]),
            "min_len": int(params[3]),
            "no_repeat_ngram_size": int(params[4])
        }

    if args.retain_dropout.lower() == "true":
        print("Generation with rataining dropout")
        EVAL_KWARGS["retain_dropout"] = True
    else:
        print("Generation without retaining dropout")

    if not args.test_file:
        raise ValueError("Missing test file")

    if not Path(args.output_file).parent.exists():
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    if Path(args.output_file).exists():
        warnings.warn("The output file will be covered", UserWarning)
    
    bs = args.batch_size
    print("----- Inference with", args.ckpt_dir, args.ckpt_file, "batch size:", bs)
    print("----- Inference with", args.test_file, "will saved as", args.output_file)
    print("----- Inference with loaded ckpt data", args.load_ckpt_data)

    pred_ckpt(
        args.test_file, args.output_file, args.ckpt_dir, args.ckpt_file, bs, args.load_ckpt_data,
        args.encoder_attn_temp, args.decoder_attn_temp, args.cross_attn_temp,
        **EVAL_KWARGS
    )

    # bs_list = [1, 16, 64]
    # bs_list = [32]
    # count = 0
    # for ckpt in ckpt_list:
    #     for bs in bs_list:
    #         print("---" * 10)
    #         count += 1
    #         print(count, "ckpt: ", ckpt, "batch_size:", bs)
    #         pred(args.test_file, args.output_file, ckpt, bs, load_ckpt_data)
