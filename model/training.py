import logging 
import os 

from tqdm import trange
import tensorflow as tf 
from model.utils import save_dict_to_json
from model.evaluation import evaluate_sess

def train_sess(sess, model_spec, num_steps, writer, params):
    '''Train the model on num_steps batches

    sess: current session
    model_spec: contain graph operations or nodes needed for training
    num_steps: (int) train for this number of batches
    params: hyperarameters
    '''
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])
    
    t = trange(num_steps)
    for i in trange(num_steps):
        if i%params.save_summary_steps ==0:

            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                summary_op, global_step])
            writer.add_summary(summ, global_step_val)
        else:

            _, _, loss_val = sess.run([train_op, update_metrics, loss])
            t.set_postfix(loss='{:05.3f}'.format(loss_val))


    metrics_values = {K: v[0] for k,v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = ' ; '.join('{} : {:05.3f}'.format(k,v) for k,v in metrics_val.items())
    logging.info('- Train metrics: '+metrics_string)

def train_and_evaluation(train_model_spec, eval_model_spec, model_dir, params, restore_from=None):
    ''' Args:
            train_model_spec: (dict) contain graph and operation or nodes needed for training
            eval_model_spec: (dict) contain graph and operation or nodes needed for evaluation
            model_dir: path contain trained model
            restore_from: (string) dir or file contain weights to restore the graph
    '''

    last_saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=1)
    begin_epoch = 0

    with tf.Session as sess:
        sess.run(train_model_spec['variable_init_op'])
        if restore_from not None:
            logging.info('Restore parameter from {}'.format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summary'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summary'), sess.graph)

        best_eval_acc= 0.0
        for epoch in range(begin_epoch, begin_epoch+params.num_epochs):
            logging.info('Epoch {}/{}'.format(epoch+1, begin_epoch +params.num_epochs))
            num_steps = (params.train_size + params.batches_size -1) // params.batches_size
            metrics = evaluate_sess(sess, eval_model_spec, num_step, eval_writer)

            # If best_val, best_save path

            eval_acc = metrics['accuracy']
            if eval_acc >=best_eval_acc:
                best_eval_acc = eval_acc
                # Save weights 
                best_save_path = os.path.join(model_dit , 'best_weights', 'after_epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch+1)
                logging.info('- Found new best accuracy, saving in {}'.format(best_save_path))
                # save best eval metrics
                best_json_path = os.path.join(model_dir, 'metrics_eval_best_weights.json')
                save_dict_to_json(metrics, best_json_path)
            
            last_json_path = os.path.join(model_dir, 'metrics_eval_last_weights.json')
            save_dict_to_json =(metrics, last_json_path)
