import tensorflow as tf 


classes = ['mot', 'hai', 'ba', 'cho']

# cast = tf.string_to_number(A)

# with tf.Session() as sess:
#     print(sess.run(cast))
class_indice = dict(zip(classes, range(len(classes))))
print(class_indice)