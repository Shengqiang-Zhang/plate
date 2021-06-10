import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", type=str)
    parser.add_argument("--ref", default="", type=str)
    parser.add_argument("--output", default="", type=str)
    # parser.add_argument("--script", default="evaluate_sum.sh", type=str)

    args = parser.parse_args()
    return args


def main(args):
    with open(args.input, 'r', encoding="utf-8") as fin, \
            open(args.ref, 'r', encoding="utf-8") as fref, \
            open(args.output, 'w', encoding="utf-8") as fout:
        for i, r in zip(fin, fref):
            len_ref = len(r.strip().split(' '))
            len_in = len(i.strip().split(' '))
            if len_in > len_ref:
                out = i.strip().split(' ')[:len_ref]
                out = ' '.join(out)
            else:
                out = i.strip()
            print(out, file=fout)


if __name__ == "__main__":
    args = get_args()
    main(args)
