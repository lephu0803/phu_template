import tensorflow as tf 
from PIL import Image

def build_model(is_training, inputs, params):
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.images_size, params.image_size, params.channels]
    
    out = images
    
    # TODO: Create VGG16 here



    return logits