import argparse
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from openvino import save_model
import os

def main():
    parser = argparse.ArgumentParser(description="Конвертация токенизатора Hugging Face в OpenVINO")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,  # Теперь скрипт не упадет, а потребует ввод
        help="Название папки модели внутри ./models/"
    )
    
    args = parser.parse_args()    
    model_dir = f"./models/{args.model_name}/1"
    hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)

    save_model(ov_tokenizer, os.path.join(model_dir, "openvino_tokenizer.xml"))
    save_model(ov_detokenizer, os.path.join(model_dir, "openvino_detokenizer.xml"))
    
    print(f"Готово! Токенизаторы сохранены в {model_dir}")

if __name__ == "__main__":
    main()
