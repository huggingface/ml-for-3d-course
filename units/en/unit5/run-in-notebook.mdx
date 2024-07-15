# Run in notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/dylanebert/ml-for-3d-course-notebooks/blob/main/capstone.ipynb)

The first open is to run in a Colab notebook. This is the easiest way to validate the code quickly.

## Setup

1. Click the "Open in Colab" button above.
2. Change the runtime type to GPU.
3. Scroll down to the `Run in this notebook section`.

## Run the demo

Start by installing dependencies.

```bash
!pip install -r https://huggingface.co/spaces/dylanebert/LGM-tiny/raw/main/requirements.txt
```

Then, run the demo code. This is exactly the same as in the space `app.py`. To ensure your model is working as expected, replace both instances of `dylanebert/LGM-full` with your `{username}/{model_name}`. Then, run the code.

```python
import shlex
import subprocess

import gradio as gr
import numpy as np
import torch
from diffusers import DiffusionPipeline

subprocess.run(
    shlex.split(
        "pip install https://huggingface.co/spaces/dylanebert/LGM-mini/resolve/main/wheel/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"
    )
)

pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")


def run(image):
    input_image = np.array(image, dtype=np.float32) / 255.0
    splat = pipeline(
        "", input_image, guidance_scale=5, num_inference_steps=30, elevation=0
    )
    splat_file = "/tmp/output.ply"
    pipeline.save_ply(splat, splat_file)
    return splat_file


demo = gr.Interface(
    fn=run,
    title="LGM Tiny",
    description="An extremely simplified version of [LGM](https://huggingface.co/ashawkey/LGM). Intended as resource for the [ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course/unit0/introduction).",
    inputs="image",
    outputs=gr.Model3D(),
    examples=[
        "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
    ],
    cache_examples=True,
    allow_duplication=True,
)
demo.queue().launch()
```

## Demo breakdown

Let's break down the demo code.

### Import dependencies

Import the required libraries.

```python
import shlex
import subprocess

import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import DiffusionPipeline
```

### Install diff-gaussian-rasterization

For the gaussian splatting step of LGM, we need to install a custom wheel. This is a workaround for the space to run on [ZeroGPU](https://huggingface.co/zero-gpu-explorers).

```python
subprocess.run(
    shlex.split(
        "pip install https://huggingface.co/spaces/dylanebert/LGM-mini/resolve/main/wheel/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"
    )
)
```

### Construct the pipeline

Construct the [LGM](https://huggingface.co/dylanebert/LGM-full) pipeline. Replace `dylanebert/LGM-full` with your `{username}/{model_name}`.

```python
pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
```

### Define the run function

Define the run function that takes an image and returns a ply file.

1. Convert the image to a numpy array and normalize it to [0, 1].
2. Run the pipeline with the default parameters.
3. Save the ply file to `/tmp/output.ply`.
4. Return the ply file.

```python
@spaces.GPU
def run(image):
    input_image = np.array(image, dtype=np.float32) / 255.0
    splat = pipeline(
        "", input_image, guidance_scale=5, num_inference_steps=30, elevation=0
    )
    splat_file = "/tmp/output.ply"
    pipeline.save_ply(splat, splat_file)
    return splat_file
```

### Create the demo

Create the demo using [Gradio](https://www.gradio.app/guides/quickstart), which handles the UI for us.

```python
demo = gr.Interface(
    fn=run,
    title="LGM Tiny",
    description="An extremely simplified version of [LGM](https://huggingface.co/ashawkey/LGM). Intended as resource for the [ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course/unit0/introduction).",
    inputs="image",
    outputs=gr.Model3D(),
    examples=[
        "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
    ],
    cache_examples=True,
    allow_duplication=True,
)
demo.queue().launch()
```

![LGM Tiny Demo](https://huggingface.co/datasets/dylanebert/ml-for-3d-course/resolve/main/lgm-tiny.png)
