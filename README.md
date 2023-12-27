# 112-1 ADL_final_project Group33 
## Abstract
We have developed a project that processes images to generate a piece of music and a poem,capturing the essence of the image. We referto it as ShotGMP — "Take a shot to Generate Music & Poem." Our approach involves combining three models into a pipeline to generate poems and music from images. Furthermore, we have prepared additional datasets to enhance the capabilities of our pipeline. The core of this project is to deliver a project capable of producing high-quality poems and music that suit the given image, while maintaining a model size reasonable enough for most people to run withease.

## Work-Flow
This is the workflow for ShotGMP. We first take some pictures, then use these pictures as input for InstructBLIP. InstructBLIP will output "Paragraph (in English)" and "Suitable Prompts" for Llama-2 and MusicGen respectively. Lastly, Llama-2 will output our poem and MusicGen will generate the music for the picture based on the InstructBLIP's input.

<img width="608" alt="from_canva" src="https://github.com/weiiiii0622/112-1_ADL_Final/assets/69110733/eec919e0-c770-4381-a7e3-cd307707523e">


## - InstructBLIP

### Folder Structure

``` bash
├── README.md
├── output
│   ├── InstructBlip_output.jsonl
│   ├── InstructBlip_ver1.json
│   ├── InstructBlip_ver2.json
│   ├── InstructBlip_ver2_test.json
│   ├── InstructBlip_ver2_train.json
│   ├── blip2_gen_stanford.json
│   └── blip2_output.json
├── InstructBlip_inference.py
├── InstructBlip_run.sh
└── stanford_dataset.csv
```

### Environment setup
We use the HuggingFace interface:
https://huggingface.co/docs/transformers/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration
```
pip install torch
pip install transformers
pip install accelerate
pip install bitsandbytes
pip install Pillow
pip install scipy
pip install jsonlines
```

### How to run
Change the arguments in `InstructBlip_run.sh` and execute it.
```bash
// InstructBlip_run.sh
export CUDA_VISIBLE_DEVICES=1
python3 InstructBlip_inference.py \
  --model_path Salesforce/instructblip-flan-t5-xl \
  --data_path "YOUR .csv DATA PATH" \
  --column_name "YOUR IMAGE COLUMN NAME IN .csv FILE" \
  --output_path "YOUR OUTPUT .json PATH"
```

## MusicGen
### Folder Structure
``` bash
├── README.md
└── musicgen.ipynb
```
### How to Run
To employ MusicGEN, plesae check [MusicGen README](./musicgen/README.md).

## GPT4-API Generated Data
### Folder Structure
``` bash
├── README.md
├── raw
│   ├── README.md
│   ├── gpt4_1k_1.json
│   └── gpt4_1k_2.json
├── gen.ipynb
├── gp4_1434.json
├── prompt.ipynb
└── prompt_gpt4_1434.json
```
### How to Run
To demonstrate how we generate data via GPT4-API, see [gen.ipynb](./gpt_data/gen.ipynb).

## Llama2 Fine-tune
### Folder Structure
``` bash
├── Poem_Demo
│   ├── gpt_prompt_demo_rhyme_words.json
│   ├── gpt_prompt_demo_rhyme.json
│   ├── gpt1epoch.json
│   ├── gpt2epoch.json
│   ├── gpt5epoch.json
│   ├── gpt10epoch.json
│   ├── gptAndTaiwan.json
│   ├── gptTranslate_2epoch.json
│   ├── gpt1334AndLLAMA.json
│   └── steps_evaluation.json
├── Source
│   ├── config_tranlate.yml
│   ├── convert_llama_weights_to_hf.py
│   ├── mergedPoemDataset.json
│   ├── dataToTune.py
│   └── test.json
├── Poem_Standard
│   ├── output_poem_gpt.json
│   ├── output_poem_llama_chinese_0_14.json
│   ├── output_poem_llama_chinese.json
│   └── output_poem.json
└── README.md

```
### How to Run
To demonstrate how we fine-tune Llama2, see [Llama README](./llama/README.md).