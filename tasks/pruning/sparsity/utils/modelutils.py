import re
import sys

sys.path.append('yolov5')

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for x, y in dataloader:
        preds.append(torch.argmax(model(x.to(dev)), 1))
        ys.append(y.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('%.2f' % acc)
    if model.training:
        model.train()

@torch.no_grad()
def test_opt(model, device, dataset, tokenizer):
    model.eval()
    # The task is to predict the last word of the input.
    for batch in dataset["text"]:
        with torch.no_grad():
            batch = tokenizer(
                batch, 
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=True
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            if not isinstance(batch, list):
                batch = [batch['input_ids'].to(device)]
            loss = model(batch[0].to(device)).loss
    return loss

def get_test(name):
    if 'opt' in name:
        return test_opt
    return test

def run_opt(model, batch):
    dev = next(iter(model.parameters())).device
    out = model(batch.to(dev))
    if isinstance(out, dict) and out.get("logits") is not None:
        return out["logits"]
    return out

def get_hf_model(model_name, cache_dir=None):
    def tokenize_function(examples):
        example = tokenizer(examples['text'])
        return example

    if "opt" in model_name.lower():
        # model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
        model = OPTForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model.eval()

    elif "llama" in model_name.lower():
        # tokenizer class name in the repo is wrong
        # the current fix is to clone the repo, modify the class name and then load the model from local
        # tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        # model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
    
    model.seqlen = model.config.max_position_embeddings

    return model, tokenizer, tokenize_function

def firstlast_names(model):
    if 'rn' in model:
        return ['conv1', 'fc']
    if 'bertsquad' in model:
        return [
            'bert.embeddings.word_embeddings',
            'bert.embeddings.token_type_embeddings',
            'qa_outputs'
        ]

def get_model_memory(model):
    """Get sizes of all parameters in `model` in Gigabytes."""
    mem = 0
    for param in model.parameters():
        data_format = int(re.findall("\d+", str(param.data.dtype))[-1])
        num_params = torch.numel(param)
        mem += num_params * data_format / (8 * 1024 ** 3)   # convert to Gigabytes
    
    return mem

def get_free_memory():
    """Get free GPU memory in Gigabytes."""
    mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    return mem

def update_module_weights(model, mod_name, updated_weights):
    mod = model.get_submodule(mod_name)
    mod.weight = Parameter(updated_weights, requires_grad=mod.weight.requires_grad)
    