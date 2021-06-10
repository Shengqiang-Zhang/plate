import argparse, os, sys, torch
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--decode_script', type=str)

    args = parser.parse_args()

    return args


def parse_ckpt_list(ckpt_dir):
    ckpt_list = []
    for file in Path(ckpt_dir).iterdir():
        if file.suffix == ".pt":
            ckpt_list.append(file)
    return ckpt_list


def run_cmd(cmd):
    os.system(cmd)


def inference_multi_gpu(n_gpus, ckpt_list, args):
    decode_script = args.decode_script
    python = sys.executable
    print(python)
    print(decode_script)

    import multiprocessing
    for i in range(0, len(ckpt_list), n_gpus):
        cur_ckpt_list = ckpt_list[i:i + n_gpus]
        pool = multiprocessing.Pool(n_gpus)

        for gpu_id, ckpt_file in enumerate(cur_ckpt_list):
            ckpt_file = Path(ckpt_file).name
            cmd = '''CUDA_VISIBLE_DEVICES={gpu_id} {python} -u {script} \
                --dataset {dataset} \
                --train_dir {train_dir} \
                --ckpt_file {ckpt_file} \
            '''.format(
                gpu_id=gpu_id,
                python=python,
                script=decode_script,
                dataset=args.dataset,
                train_dir=args.train_dir,
                ckpt_file=ckpt_file,
            )

            print(cmd)
            pool.apply_async(run_cmd, args=(cmd,))

        pool.close()
        pool.join()


def main():
    args = get_args()
    args.ckpt_list = parse_ckpt_list(args.ckpt_dir)
    print(args.ckpt_list)
    n_gpus = min(args.n_gpus, len(args.ckpt_list))
    inference_multi_gpu(n_gpus, args.ckpt_list, args)


if __name__ == '__main__':
    main()
