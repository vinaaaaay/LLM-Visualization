import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def download_smol():
    # SmolLM2 is state-of-the-art for tiny models
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    save_directory = "./smollm_local"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print(f"--- Downloading {model_name} (Ultra-lightweight) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    print(f"\nSuccess! Saved in '{save_directory}'.")
    print("This will be significantly faster on your 8GB M2.")


if __name__ == "__main__":
    download_smol()

