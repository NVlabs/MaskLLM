# for i from 00000 to 00100
mkdir -p assets/data/preprocessed
for i in {00000..00019}; do
       echo "Processing ./assets/data/en/c4-train.${i}-of-01024.json"
       python tools/preprocess_data.py \
              --input "./assets/data/en/c4-train.${i}-of-01024.json" \
              --output-prefix assets/data/preprocessed/llama2_${i} \
              --vocab-file ./assets/checkpoints/llama2_7b_hf/tokenizer.json \
              --tokenizer-type Llama2Tokenizer \
              --tokenizer-model ./assets/checkpoints/llama2_7b_hf/tokenizer.model \
              --append-eod \
              --workers 8
done