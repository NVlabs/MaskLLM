# for i from 00000 to 00100
mkdir -p assets/data/c4_llama3.1_pretokenized
for i in {00000..00019}; do
       echo "Processing ./assets/data/c4/en/c4-train.${i}-of-01024.json"
       python tools/preprocess_data.py \
              --input "./assets/data/c4/en/c4-train.${i}-of-01024.json" \
              --output-prefix assets/data/c4_llama3.1_pretokenized/c4_llama3.1_${i} \
              --vocab-file ./assets/checkpoints/llama3.1_8b_hf/tokenizer.json \
              --tokenizer-type AutoTokenizer \
              --tokenizer-model ./assets/checkpoints/llama3.1_8b_hf \
              --append-eod \
              --workers 8
done