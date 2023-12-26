# 112-1 ADL_final_project Group33
## Abstract
We have developed a project that processes images to generate a piece of music and a poem,capturing the essence of the image. We referto it as ShotGMP — "Take a shot to Generate Music & Poem." Our approach involves combining three models into a pipeline to generate poems and music from images. Furthermore, we have prepared additional datasets to enhance the capabilities of our pipeline. The core of this project is to deliver a project capable of producing high-quality poems and music that suit the given image, while maintaining a model size reasonable enough for most people to run withease.


## - Stage1

### BLIP-2

#### Huggingface Env Setup
- Link: https://huggingface.co/docs/transformers/main/model_doc/blip-2#transformers.Blip2ForConditionalGeneration
- Create virtual environment
    ```bash=
    virtualenv ADL
    ```
- Install packages
    - Option 1 
    ```bash=
    pip install -r BLIP2_requirements.txt
    ```
    - Option2 (manually)
    ```bash=
    pip install torch
    pip install transformers
    pip install accelerate
    pip install bitsandbytes
    pip install Pillow
    pip install scipy
    ```
