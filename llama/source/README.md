### Download llama model
* Download model by following instructions in the link the link : [link to download](https://github.com/facebookresearch/llama?fbclid=IwAR0NnxiermvwPeNBvH3IthZZKPM99YCO_OE5_Ol5F3lIIGlQxb-mKmJJiVU). 
(clone repo -> install environment -> request download link -> download llama-2-7b or other model)
* The model need to be converted to the Hugging Face Transformers format using the conversion script: [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) with the command: 
```bash    
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```
### Finetune
* Tune model with axolotl. Reference: [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'
```

#### Usage
```bash
# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./lora-out"
    
# Replace .yml file with your configuration file.

```
