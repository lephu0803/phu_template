import logging
import os 
import tensorflow as tf 
from model.model_fn import model_fn
import tqdm
from utils import save_dict_to_json 

def evaluate_sess(sess, model_spec, num_steps, writer = None, params=None):
    '''Train the model on num_steps batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contain the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        parrams: (params) hyperparameters
    '''

    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    ### Load the evaluation set to the data pineline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    for _ in range(num_steps):
        sess.run(update_metrics)

    metrics_value = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_value) 
    metrics_string = " ; ".join('{}: {:05.3f}'.format(k,v) for k, v in metrics_value.items())
    logging.info('- Eval metrics:' +metrics_string)

    # Add summaries manually to writer at global_step_val 
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val

def evaluate(model_spec, model_dir, params, restore_from):
    '''Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weight and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, svae_summary_steps

        resotre_from: (strings) directory restore from file containing weights and graph

    '''

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(model_spec['variables_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        #Evaluate

        num_steps = (params.eval_size + params.batch_size - 1) //params.batch_size
        metrics = evaluate_sess(sess, model_spec, num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(metrics_name))
        save_dict_to_json(metrics, save_path)
        