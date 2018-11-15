import numpy as np 
import tensorflow as tf 
from PIL import Image
import cv2 

class Data_Generator():
    def __init__(self, image_size):
        self.IMAGE_SIZE = image_size
    def tf_image_resize(self, image_file_path):
        # tf.reset_default_graph()
        X_data = []
        X = tf.placeholder(tf.float32, (None, None, 3))
        tf_img = tf.image.resize_images(X, [self.IMAGE_SIZE, self.IMAGE_SIZE],
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)   
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for index, file_path in enumerate(X_image_file_path):
                img = Image.open(file_path)[:,:,:3]
                resized_img = sess.run(tf_img, feed_dict = {X: img}) 
                X_data.append(resized_img)
        X_data = np.array(X_data, dtype = np.float32)
        return X_data

    def __central_scaling(self, input_images, scales):
        
        assert scales == None, "Need define parameter for scaling with array format"
        boxes = np.zeros((len(scales), 4), dtype = np.float32)
        for index, scale in enumerate(scales):
            x1 = y1 = 0.5 -0.5 * scale
            x2=y2 = 0.5+0.5*scale 
            boxes[index]= np.array([y1,x1,y2,x2], dtype = np.float32)
        box_ind = np.zeros((len(scales)), dtype = np.int32)
        crop_size = np.array([self.IMAGE_SIZE, self.IMAGE_SIZE], dtype = np.int32)

        X_scaled_data = []
        X = tf.placeholder([1,self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
        with tf.Session as sess:
            for img_data in input_images:
                batch_image = np.expand_dims(img_data, axis = 0)
                scaled_img = sess.run(tf_img, feed_dict = {X:img_data})
                X_scaled_data.extend(scaled_img)
        X_scaled_data = np.array(X_scaled_data)
        return X_scaled_data

    def __rotate(self, start_angle, end_angle, n_images):
        X_rotate = []
        iterate_at =(end/angle - start_angle)/(n_images-1)

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(None, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        radian = tf.placeholder(tf.float32, shape(len(X_imgs)))
        tf_img = tf.contrib.image.rotate(X,radian)
        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())

            for index in range(n_images):
                degrees_angle = start_angle + index*iterate_at
                radian_value = degrees_angle*pi/180
                radian_arr = [radian_value]*len(X_imgs)
                rotated_imgs = sess.run(tf_img, feed_dict = {X:X_imgs, radian:radian_arr})
                X_rotate.append(rotated_imgs)
            X_rotate = np.array(X_rotate, dtype = np.float32)
            return X_rotate

        def flip_image(self, X_imgs):
            X_flip = []
            tf.reset_default_graph()
            X = tf.placeholder(tf.float32, (self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
            tf_img1 = tf.image.flip_left_right(X)
            tf_img2 = tf.image.flip_up_down(X)
            tf_img3 = tf.image.transpose_image(X)
            with tf.Session as sess:
                sess.run(tf.global_variables_initializer())
                for img in X_imgs:
                    flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X:img})
                    X_flip.extend(flipped_imgs)
            X_flip = np.array(X_flip, dtype = np.float32)
            return X_flip

        def add_noise(self, X_imgs, salt_pepper_ratio = 0.2, amount= 0.4):
            X_imgs_copy = X_imgs.copy()
            row, col, _ = X_imgs_copy[0].shape

            num_salt = np.cell(amount*X_imgs_copy[0].size*salt_pepper_ratio)
            num_pepper = np.cell(amount*X_imgs_copy[0].size*(1.0-salt_pepper_ratio))

            for X_img in X_imgs_copy:
                # Add salt
                coords = [np.random.randint(0,i-1,int(num_salt)) for i in X_img.shape]
                X_img[coords[0], coords[i], :] = 1

                #Add Pepper noise
                coords = [np.random.randint(0, i -1, int(num_pepper)) for i in X_img.shape]
                X_img[coords[0], coords[1], :] = 0
            return X_imgs_copy

        def gaussian_noise(self, X_imgs):
            gaussian_imgs = [] 
            row, col, _ = X_imgs[0].shape

            mean = 0
            var =0.1

            sig = var **0.5

            for X_img in X_imgs:
                gaussian = np.random.random((row, col, 1)).astype(np.float32)
                gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
                gaussian_img = cv2.addWeight(gaussian_img)
                gaussian_imgs.append(gaussian_img)
            gaussian_imgs = np.array(gaussian_imgs, dtype= np.float32 )
            return gaussian_imgs 
 

