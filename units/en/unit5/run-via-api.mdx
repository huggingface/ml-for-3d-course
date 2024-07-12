# Run via API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/dylanebert/ml-for-3d-course-notebooks/blob/main/capstone.ipynb)

<iframe src="https://dylanebert-LGM-tiny-api.hf.space" frameborder="0" width="850" height="450"></iframe>

To run via API, instead of duplicating the [LGM-tiny](https://huggingface.co/spaces/dylanebert/LGM-tiny) space, duplicate the [LGM-tiny-api](https://huggingface.co/spaces/dylanebert/LGM-tiny-api) space. This contains the following `app.py`.

```python
import gradio as gr
from gradio_client import Client, file


def run(image_url):
    client = Client("dylanebert/LGM-tiny")
    image = file(image_url)
    result = client.predict(image, api_name="/predict")
    return result


demo = gr.Interface(
    fn=run,
    title="LGM Tiny API",
    description="An API wrapper for [LGM Tiny](https://huggingface.co/spaces/dylanebert/LGM-tiny). Intended as a resource for the [ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course).",
    inputs=gr.Textbox(label="Image URL", placeholder="Enter image URL, e.g. https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"),
    outputs=gr.Model3D(),
    examples=[
        "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
    ],
    allow_duplication=True,
)
demo.queue().launch()
```

This will work on CPU, but relies on the original LGM-tiny, instead of your custom model. However, is your focus is on UI/UX or downstream tasks, this may be acceptable.
