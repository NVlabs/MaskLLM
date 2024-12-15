import torch
import argparse

parser = argparse.ArgumentParser(description='Trim Lana checkpoint')
parser.add_argument('--ckpt_dir', type=str, default='output/checkpoints/llama-mask-only/train_iters_2000/ckpt/iter_0002000', help='Input checkpoint')
args = parser.parse_args()

def trim_ckpt(input, output):
    ckpt = torch.load(input, map_location='cpu')
    new_encoder_state_dict = {}
    mask_options = torch.zeros(1, 6, 4, dtype=torch.float32)
    mask_options[:, 0, :].data += torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    mask_options[:, 1, :].data += torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    mask_options[:, 2, :].data += torch.tensor([1, 0, 0, 1], dtype=torch.float32)
    mask_options[:, 3, :].data += torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    mask_options[:, 4, :].data += torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    mask_options[:, 5, :].data += torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    for k,v in ckpt['model']['language_model']['encoder'].items():
        if '.diff_mask.gate' in k: 
            gate = ckpt['model']['language_model']['encoder'][k].float()
            runtime_mask = ckpt['model']['language_model']['encoder'][k.replace('diff_mask.gate', 'mask')].float()
            winner_mask = mask_options[torch.arange(mask_options.shape[0]), gate.argmax(dim=-1)].view(*runtime_mask.shape)
            # set the type of winner mask the same as runtime_mask
            winner_mask = winner_mask.type_as(runtime_mask)
            new_encoder_state_dict[k.replace('diff_mask.gate', 'mask')] = winner_mask
            print("save winner mask:", k.replace('diff_mask.gate', 'mask'))
            continue

        if '.mask' in k: continue
        if '.mask_options' in k: continue

        new_encoder_state_dict[k] = v
        
    ckpt['model']['language_model']['encoder'] = new_encoder_state_dict
    print(ckpt['model']['language_model']['encoder'].keys())
    torch.save(ckpt, output)


import os
import glob

# Create output directory
splited_dir = args.ckpt_dir.split('/') 
output_dir = os.path.join('/'.join(splited_dir[:-1]), 'release')
print(f"output_dir: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Trim the checkpoints
mp_rank_dirs = glob.glob(os.path.join(args.ckpt_dir, "mp_rank_*"))
for mp_rank_dir in mp_rank_dirs:
    ckpt_file = os.path.join(mp_rank_dir, "model_optim_rng.pt")
    output_file  = ckpt_file.replace(args.ckpt_dir, output_dir)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Trim {ckpt_file} to {output_file}")
    trim_ckpt(ckpt_file, output_file)

# update the latest iteration to "release"
iteration_file = os.path.join( *splited_dir[:-1], 'latest_checkpointed_iteration.txt')
print(iteration_file)
with open(iteration_file, 'w') as f:
    f.write("release")

            
