# MusicGen Environment Setup
Reference: https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
## Prerequisites
- Have to first install AudioCraft
- AudioCraft requires Python 3.9, PyTorch 2.0.0.
- It is recommended to install ffmpeg to deal with audio files.
- It is recommended to use GPU with 16GB of RAM to run "facebook/musicgen-medium" or "facebook/musicgen-melody" (size of 1.5B).
    - small/medium/big supports text -> music. melody supoorts text/text+music -> music
    - The small/medium/big version support transformer library, and melody only uses MusicGen library.
## Installation
```shell
python -m pip install -U audiocraft 
```
## Usage
Reference:
1. https://huggingface.co/docs/transformers/main/en/model_doc/musicgen
   - The usage of transformer library. (Doesn't support melody version)
2. https://github.com/facebookresearch/audiocraft/blob/main/demos/musicgen_demo.ipynb
   - Direct usage of MusicGen libray. 