# Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/dylanebert/ml-for-3d-course-notebooks/blob/main/multi_view_diffusion.ipynb)

In our case, we'll be using a pretrained pipeline:

```python
import torch
from diffusers import DiffusionPipeline

multi_view_diffusion_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/multi-view-diffusion",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
```

The name of the model is [dylanebert/multi-view-diffusion](https://huggingface.co/dylanebert/multi-view-diffusion), a mirror of [ashawkey/mvdream-sd2.1-diffusers](https://huggingface.co/ashawkey/mvdream-sd2.1-diffusers). For any pretrained model, you can find the model card on the Hugging Face Hub at `https://huggingface.co/<model-name>`, which contains information about the model.

In our case, we also need to load the custom pipeline (also at `dylanebert/multi-view-diffusion`) to use the model. This is because diffusers doesn't officially support 3D. So, for the purposes of this course, I've wrapped the model in a custom pipeline that allows you to use it for 3D tasks.

## Load an Image

```python
import requests
from PIL import Image
from io import BytesIO


image_url = "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/inputs/images/a_cat_statue.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image
```

![Cat Statue](https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/inputs/images/a_cat_statue.jpg)

With this code, we load and display the famous [Cat Statue](https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/inputs/images/a_cat_statue.jpg), used for image-to-3D demos.

## Run the Pipeline

```python
import numpy as np

def create_image_grid(images):
    images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

    width, height = images[0].size
    grid_img = Image.new("RGB", (2 * width, 2 * height))

    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))

    return grid_img

image = np.array(image, dtype=np.float32) / 255.0
images = multi_view_diffusion_pipeline("", image, guidance_scale=5, num_inference_steps=30, elevation=0)

create_image_grid(images)
```

Finally, we run the pipeline on the image.

The `create_image_grid` function isn't part of the pipeline. It's just a helper function to display the results in a grid.

To run the pipeline, we simply prepare the image by converting it to a normalized numpy array:

`image = np.array(image, dtype=np.float32) / 255.0`

Then, we pass it to the pipeline:

`images = multi_view_diffusion_pipeline("", image, guidance_scale=5, num_inference_steps=30, elevation=0)`

Where parameters `guidance_scale`, `num_inference_steps`, and `elevation` are specific to the multi-view diffusion model.

![Multi-view Cats](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ml-for-3d-course/multi-view-cats.png)

## Conclusion

Congratulations! You've run a multi-view diffusion pipeline.

Now what about hosting your own demo?
