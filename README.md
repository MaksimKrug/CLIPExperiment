# OpenAI CLIP Experiment
Experiment with OpeanAI [CLIP](https://github.com/openai/CLIP) model for the number of men and women in the photo prediction

## Solution
[CLIP](https://github.com/openai/CLIP) used to calculate similarity between image and text inputs and return the closest result. For query used: "There are WOMEN_NUMBER and MEN_NUMBER in the photo" for different predifined combinations of men and women in the photo.

Common questions:
1. <b>Q:</b> Why the model version is "ViT-L/14"? <b>A:</b> For the first glimpse work better than standard ViT-B/32.
2. <b>Q:</b> Why qeries for men and women are not separated? <b>A:</b> Model often do mistakes when trying to identify the gender and for the image with 2 men and 1 woman easily can return something like: "3 men" and "3 women" in the photo. I decided to use more queries, which helped to make the model more stable regards to the number of persons prediction.

## Start Docker
```
docker build --rm -t clip_experiment ./
docker run --rm -v $(pwd)/examples:/workdir/examples clip_experiment --image_path=examples/IMAGE_PATH
```
