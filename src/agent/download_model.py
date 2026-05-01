from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "tiiuae/falcon-7b-instruct"

# Short path (correct)
cache_dir = r"D:\models"

print("Creating folder...")
os.makedirs(cache_dir, exist_ok=True)

print("⬇️ Downloading FULL tokenizer...")
AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    force_download=True
)

print("⬇️ Downloading FULL model (this is large ~14GB)...")
AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    force_download=True
)

print("✅ DONE. Model fully downloaded.")