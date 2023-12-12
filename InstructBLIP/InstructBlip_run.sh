export CUDA_VISIBLE_DEVICES=1
python3 InstructBlip_inference.py \
  --model_path Salesforce/instructblip-flan-t5-xl \
  --data_path ./stanford_dataset.csv \
  --column_name url \
  --output_path ./InstructBlip_ver1.json