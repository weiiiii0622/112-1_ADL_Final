# 112-1 ADL_final_project Group33 
## Abstract
We have developed a project that processes images to generate a piece of music and a poem,capturing the essence of the image. We referto it as ShotGMP — "Take a shot to Generate Music & Poem." Our approach involves combining three models into a pipeline to generate poems and music from images. Furthermore, we have prepared additional datasets to enhance the capabilities of our pipeline. The core of this project is to deliver a project capable of producing high-quality poems and music that suit the given image, while maintaining a model size reasonable enough for most people to run withease.


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