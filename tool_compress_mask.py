import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Mask Compression')
parser.add_argument('--mask_ckpt', type=str, help='path to the mask checkpoint')
parser.add_argument('--output', type=str, help='output path')

args = parser.parse_args()

if __name__=='__main__':
    mask_ckpt = torch.load(args.mask_ckpt, map_location='cpu')
    compressed_mask = {}
    for k, mask in mask_ckpt.items():
        # Compress with np.packbits
        print(f"Compressing {k}...")
        mask = mask.cpu().numpy().astype(bool)
        mask = np.packbits(mask)
        compressed_mask[k] = mask
    np.savez_compressed(args.output, **compressed_mask)

