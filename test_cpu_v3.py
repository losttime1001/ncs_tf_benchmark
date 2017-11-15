import tensorflow as tf
import numpy as np
import cv2
import sys
import time

def run(input_image):
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./output/inception-v3.meta')
        saver.restore(sess, 'inception_v3.ckpt')

        softmax_tensor = sess.graph.get_tensor_by_name('Softmax:0')
        feed_dict = {'input:0': input_image}
        classification = sess.run(softmax_tensor, {'input:0': input_image})    #first run fo warm-up
        
        start_time = time.time()
        classification = sess.run(softmax_tensor, {'input:0': input_image})
        print 'predict label:', np.argmax(classification[0])
        print 'predict time:', time.time() - start_time, 's'

if __name__=="__main__":
    args = sys.argv
    if len(args) != 2:
        print 'Usage: python %s filename'%args[0]
        quit()
    image_data =tf.gfile.FastGFile(args[1], 'rb').read()
    image = cv2.imread(args[1])
    image = cv2.resize(image, (299,299))
    image = np.array(image)/255.0
    image = np.asarray(image).reshape((1, 299, 299, 3))
    run(image)
