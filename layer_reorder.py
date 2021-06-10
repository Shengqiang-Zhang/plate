import os, sys, argparse, torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_ckpt')
    parser.add_argument('--out_ckpt')
    parser.add_argument('--encoder-layers-to-keep')
    parser.add_argument('--decoder-layers-to-keep')

    args = parser.parse_args()

    return args


def layer_reorder(orig_ckpt_file, out_ckpt_file, encoder_layers_to_keep,
                  decoder_layers_to_keep):
    def ckptlist2nums(layers_to_keep):
        num_list = list(map(int, layers_to_keep.strip().split(',')))
        return num_list

    def get_reordered_layers(layers, layers_to_keep):
        layers_to_keep = ckptlist2nums(layers_to_keep)
        layers_left = []
        for i in layers:
            if not i in layers_to_keep:
                layers_left.append(i)

        return layers_to_keep + layers_left

    orig_ckpt = torch.load(orig_ckpt_file)
    print(type(orig_ckpt))
    model = orig_ckpt['model']

    enc_idx = set()
    dec_idx = set()
    for k, v in model.items():
        if k.startswith('encoder.layers'):
            enc_idx.add(int(k.strip().split('.')[2]))

        if k.startswith('decoder.layers'):
            dec_idx.add(int(k.strip().split('.')[2]))

    enc_layers = list(enc_idx)
    enc_layers.sort()
    dec_layers = list(dec_idx)
    dec_layers.sort()

    print(enc_layers, dec_layers)

    enc_layers_reorder = get_reordered_layers(enc_layers,
                                              encoder_layers_to_keep)
    dec_layers_reorder = get_reordered_layers(dec_layers,
                                              decoder_layers_to_keep)

    new_ckpt = {}
    for k, v in orig_ckpt.items():
        if k == 'model':
            new_model = reorder_layers(model, enc_layers_reorder,
                                       dec_layers_reorder)
            new_ckpt[k] = new_model
        else:
            new_ckpt[k] = v

    torch.save(new_ckpt, out_ckpt_file)


def reorder_layers(model, enc_layers_reorder, dec_layers_reorder):
    def get_map(layer_reorder):
        return dict([(ir, i) for i, ir in enumerate(layer_reorder)])

    new_model = {}

    enc_map = get_map(enc_layers_reorder)
    dec_map = get_map(dec_layers_reorder)

    for k, v in model.items():
        if k.startswith('encoder.layers') or k.startswith('decoder.layers'):
            fds = k.strip().split('.')
            idx = int(fds[2])
            new_idx = enc_map[idx] if k.startswith(
                'encoder.layers') else dec_map[idx]
            fds[2] = str(new_idx)
            new_k = '.'.join(fds)
            new_model[new_k] = v
        else:
            new_model[k] = v

    return new_model


if __name__ == '__main__':
    args = get_args()
    print(args)

    print('options in fairseq *****')
    nenc = len(args.encoder_layers_to_keep.strip().split(','))
    fairseq_enc_layers = ','.join(map(str, range(nenc)))
    ndec = len(args.decoder_layers_to_keep.strip().split(','))
    fairseq_dec_layers = ','.join(map(str, range(ndec)))
    print('--encoder-layers-to-keep {}'.format(fairseq_enc_layers))
    print('--decoder-layers-to-keep {}'.format(fairseq_dec_layers))
    print('*******************************************************\n\n')

    layer_reorder(args.orig_ckpt, args.out_ckpt, args.encoder_layers_to_keep,
                  args.decoder_layers_to_keep)
