# MaskLLM with C4 Dataset

## Dataset Download

Download the C4 subset 00000-00019.

```bash
python tools/download_c4.py
```

Output
```
assets/data
├── ...
└── en
    ├── c4-train.00000-of-01024.json
    ├── c4-train.00001-of-01024.json
    ├── c4-train.00002-of-01024.json
    ├── c4-train.00003-of-01024.json
    ├── c4-train.00004-of-01024.json
    ├── c4-train.00005-of-01024.json
    ├── c4-train.00006-of-01024.json
    ├── c4-train.00007-of-01024.json
    ├── c4-train.00008-of-01024.json
    ├── c4-train.00009-of-01024.json
    ├── c4-train.00010-of-01024.json
    ├── c4-train.00011-of-01024.json
    ├── c4-train.00012-of-01024.json
    ├── c4-train.00013-of-01024.json
    └── c4-train.00014-of-01024.json
    ...
```

## [Megatron Data Preprocessing](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing)

### Requirements

We use [``pytorch:24.01-py3``](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html) as the base image. Please make sure you have installed docker.

Install additional packages:
```bash
pip install nltk sentencepiece
```

### Pre-processing for LLaMA-2

```bash
bash tools/prepare_c4_megatron_llama2.py
```

```bash
assets/data/preprocessed/
├── llama2_00000_text_document.bin
├── llama2_00000_text_document.idx
├── llama2_00001_text_document.bin
├── llama2_00001_text_document.idx
├── llama2_00002_text_document.bin
├── llama2_00002_text_document.idx
├── llama2_00003_text_document.bin
├── llama2_00003_text_document.idx
├── llama2_00004_text_document.bin
├── llama2_00004_text_document.idx
...
```

To use this in Megatron-LM, we provide a blending file [assets/c4-blend.sh](assets/c4-blend.sh) for training. 

### Pre-processing for LLaMA-3

The proprecessing for LLaMA-3 is similar to LLaMA-2, but with a different script.

```bash
bash tools/prepare_c4_megatron_llama3.py
```

```bash
assets/data/preprocessed/
├── llama3_00000_text_document.bin
├── llama3_00000_text_document.idx
├── llama3_00001_text_document.bin
├── llama3_00001_text_document.idx
├── llama3_00002_text_document.bin
├── llama3_00002_text_document.idx
├── llama3_00003_text_document.bin
├── llama3_00003_text_document.idx
├── llama3_00004_text_document.bin
├── llama3_00004_text_document.idx
...
```



