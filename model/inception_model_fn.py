import tensorflow as tf 
from inception_v3 import inception_v3

def model_fn(mode, inputs, params, reuse = False):
    ''' Define graph operation 
        
        Args:   mode: define training mode or test mode
                inputs: input dictionary contain input of graph and labels
                params: contain hyperparameter of the model
                reuse: (bool) whether reuse weights
    '''
    assert (mode != 'train') or (mode != 'eval'), 'You must assign training or eval for mode field'

    is_training = (mode == 'train')
    images = inputs['images']
    labels = inputs['labels']
    num_class = len(labels) # numbers of labels
    labels = tf.cast(labels, tf.int64)

    # Define training step 
    with tf.variable_scope('train', reuse=reuse):
        logits = inception_v3(inputs=images, num_classes=num_class, is_training=True)
        prediction = tf.argmax(logits, 1)

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, prediction), tf.float32))

    if is_training:
        # Minimize with adam optimizer
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step = global_step)

    # Metric and summaries

    with tf.variable_scope('metrics'):
        metrics = {
                    'accuracy': tf.metrics.accuracy(labels=labels, predictions= tf.argmax(logits, 1)),
                    'loss': tf.metrics.mean(loss)
                }
    # group update op for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # get op to reset local variables in tf.metric

    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summary for training

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    # config
    # TODO: config evaluation
    mask = tf.not_equal(labels,prediction)

    for label, in range(0,params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(prediction, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec