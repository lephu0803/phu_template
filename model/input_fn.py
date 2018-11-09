import tensorflow as tf 

def _parse_func(filename, label, size):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize_images(image, [64,64])

    return resized_image, label

def train_preprocess(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0/255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image,0.0,1.0)

    return image, label

def input_fn(is_traning, filenames, labels, params):
    ''' This input function create specialize for SIGN dataset.abs
    The filenames have format "{label}_IMG_{id}_.jpg".
    For instance: "data_dir/2_IMG_4582.jpg". '''

    num_samples = len(filenames)

    assert len(filenames) == len(labels), "Filename and labels should have same length"

    parse_fn = lambda f, l: _parse_func(f, l, params.image)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_traning:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                    .shuffle(num_samples)
                    .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                    .map(train_fn, num_parallel_calls=params.num_parallel_calls)
                    .batch(params.batch_size)
                    .prefetch(1)
                    )

    else: 
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                    .map(parse_fn)
                    .batch(params.batch_size)
                    .prefetch(1)
                )
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {  'images': images,
                'labels': labels,
                'iterator_init_op': iterator_init_op
                }
    return inputs