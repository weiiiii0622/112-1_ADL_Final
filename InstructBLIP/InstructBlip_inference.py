from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

from tqdm.auto import tqdm
import json
import jsonlines
import argparse
import pandas as pd

def load_model(path):
    model = InstructBlipForConditionalGeneration.from_pretrained(path)
    processor = InstructBlipProcessor.from_pretrained(path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

def inference(model, processor, device, urls, prompts, args):

    result = []
    failed = []
    total = len(urls)
    if args.append:
        with open(args.prev_output_path, 'r') as json_file:
            data = json.load(json_file)

    with jsonlines.open("InstructBlip_output.jsonl", mode='a') as writer:
        for idx, url in tqdm(iterable=enumerate(urls), total=total, colour="green"):
            
            try:
                obj = {}
                if args.append:
                    obj = data[idx]
                obj['url'] = url

                img_path = "/tmp2/b10902138/.cache/stanford_images/" + url.split('/')[-1]
                image = Image.open(img_path).convert("RGB")

                for prompt in prompts:
                    inputs = processor(images=image, text=prompts[prompt], return_tensors="pt").to(device)

                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_length=256,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                    )
                    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    obj[prompt] = generated_text
                writer.write(obj)
                result.append(obj)
            except:
                failed.append(url)

    return result, failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", default=False, help="DEBUG")
    parser.add_argument("--append", action="store_true", default=False, help="APPEND")
    parser.add_argument("--model_path", type=str, help="The path for your model")
    parser.add_argument("--data_path", type=str, help="The path for data")
    parser.add_argument("--column_name", type=str, help="The path for data")
    parser.add_argument("--prev_output_path", type=str, help="The path for prev output file")
    parser.add_argument("--output_path", type=str, help="The path for output file")

    args = parser.parse_args()

    # Get model
    model, processor, device = load_model(args.model_path)

    # Get data
    data = pd.read_csv(args.data_path)
    urls = data[args.column_name]
    if args.debug:
        urls = [urls[i] for i in range(5)]

    prompts = {
        'paragraph':   "Write a detailed description for the photo",
        'vibe':        "Question: Give 3 adjectives to describe the vibe of image. Answer:",
        'time':        "Question: What time of the day does it represent? Answer:",
        'music-era':   "Question: What time era does it represent? Answer:",
        'emotion':     "Question: What emotions do you think the person taking this image is experiencing? Answer:",
        'music-style': "Question: Describe the music style. Answer:",
    }

    # Inference
    result, failed = inference(model, processor, device, urls, prompts, args)

    # Output
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False) 

    print(failed)