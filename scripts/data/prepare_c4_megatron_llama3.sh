# for i from 00000 to 00100
mkdir -p assets/data/preprocessed_llama3
for i in {00000..00019}; do
       echo "Processing ./assets/data/en/c4-train.${i}-of-01024.json"
       python tools/preprocess_data.py \
              --input "./assets/data/en/c4-train.${i}-of-01024.json" \
              --output-prefix assets/data/preprocessed_llama3/llama3_${i} \
              --vocab-file ./assets/checkpoints/llama3_8b_hf/tokenizer.json \
              --tokenizer-type AutoTokenizer \
              --tokenizer-model ./assets/checkpoints/llama3_8b_hf \
              --append-eod \
              --workers 8
done