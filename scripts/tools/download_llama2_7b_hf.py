import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
os.makedirs(f"./assets/cache", exist_ok=True)

def save_llama_model(model_id, save_directory="assets/checkpoints/llama"):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        cache_dir=f"./assets/cache", 
        use_auth_token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    # Save model and tokenizer to the specified directory
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

# Replace 'LLaMA-2-model-id' with the actual model ID from Hugging Face's model hub
dense_dir = "assets/checkpoints"
os.makedirs(dense_dir, exist_ok=True)
model_id = "meta-llama/Llama-2-7b-hf"
save_directory = f"{dense_dir}/llama2_7b_hf"
save_llama_model(model_id, save_directory)