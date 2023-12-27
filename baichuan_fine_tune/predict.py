import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from peft import PeftModel
import argparse

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"\
    以下為任務指令，輸出符合指令的回應。\n\n\
    將後面的唐詩翻譯成英文：孤鴻海上來，池潢不敢顧。\n回應：A solitary swan flew in from the sea, Passing on ponds without thinking twice.\n\n\
    將後面的唐詩翻譯成英文：幽人歸獨臥，滯慮洗孤清。\n回應：As a hermit I recline in isolation, Cleansing and clearing thoughts as reclusion intends.\n\n\
    將後面的唐詩翻譯成英文：{instruction}\n回應："

def predict(model, tokenizer, data, max_new_tokens=256):
    predictions = []
    for x in data:
        instruction = get_prompt(x["poem"])
        tokenized_instruction = tokenizer(instruction, return_tensors='pt', add_special_tokens=False)

        with torch.no_grad():
            output = model.generate(**tokenized_instruction, max_new_tokens = max_new_tokens)

        generated_text = tokenizer.decode(output[0, len(tokenizer.encode(instruction)):], skip_special_tokens=True)
        predictions.append({"poem": x["poem"], "translate": generated_text})

    output_dir = os.path.dirname(args.output_file)
    if output_dir != None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        default="baichuan-inc/Baichuan2-13B-Chat",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        default="output/adapter_model",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        default="test_data.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/translation.json",
    )
    args, unknown = parser.parse_known_args()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()
    predict(model, tokenizer, data)