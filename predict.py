import clip
import torch

import argparse
import os
import sys
from PIL import Image

from itertools import product
import json

PREFIX = "There are {WOMAN_MASK} and {MAN_MASK} in the photo"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)
model, preprocess = clip.load("ViT-L/14", device)


@torch.no_grad()
def encode_image(image_input: torch.Tensor) -> torch.Tensor:
    # Encode image
    return model.encode_image(image_input)


@torch.no_grad()
def encode_text(text_input: torch.Tensor) -> torch.Tensor:
    # Encode text
    return model.encode_text(text_input)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        prog="The number of persons in the a given photo",
        description="The program using OpeanAI CLIP to detect the number of persons",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image"
    )
    parser.add_argument(
        "--categories_path",
        type=str,
        required=False,
        default="categories.json",
        help="Path to the file with categories",
    )
    args = parser.parse_args()

    # Load image
    image_path = args.image_path
    if not os.path.exists(image_path):
        sys.exit(f"File {image_path} does not exist, check it")
    image = Image.open(image_path)

    # Load categories
    categories_path = args.categories_path
    if not os.path.exists(categories_path):
        sys.exit(f"File {categories_path} does not exist, check it")
    categories = json.load(open(categories_path, "r"))

    # Encode image
    image_input = preprocess(image).unsqueeze(0).to(device)
    image_features = encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Encode text
    queries = product(categories["Woman"], categories["Man"])
    queries = [PREFIX.format(WOMAN_MASK=q[0], MAN_MASK=q[1]) for q in queries]
    text_input = torch.cat([clip.tokenize(t) for t in queries]).to(device)
    text_features = encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    result = queries[indices[0]]
    print(result)
