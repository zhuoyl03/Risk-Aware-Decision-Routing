import torch
from transformers import CLIPForImageClassification, ViTForImageClassification


def initialize_model(model_name, num_classes, keep_frozen=False, use_pretrained=True):
    model = None
    input_image_size = 0
    resize_image_size = 0
    if model_name == "vit":
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                            num_labels=num_classes,
                                                            )
        input_image_size = 224
        resize_image_size = 256

    elif model_name == "clip":
        model = CLIPForImageClassification.from_pretrained('openai/clip-vit-base-patch32',
                                                          num_labels=num_classes,
                                                          )
        input_image_size = 224
        resize_image_size = 256
    
    else:
        print("Invalid model name, exiting...", flush=True)
        exit()

    return model, resize_image_size, input_image_size