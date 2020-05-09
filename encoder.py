import tensorflow as tf
import numpy as np
import cv2


def extract_person(image, bbox, person_image_shape):
    bbox = np.array(bbox)
    new_width = float(person_image_shape[1]) / person_image_shape[0] * bbox[3]
    bbox[0] -= (new_width - bbox[2]) / 2
    bbox[2] = new_width
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image = cv2.resize(image, tuple(person_image_shape[::-1]))
    return image


class Encoder(object):
    def __init__(self, checkpoint_filename):
        self.sess = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")

        self.input_var = tf.get_default_graph().get_tensor_by_name("net/images:0")
        self.output_var = tf.get_default_graph().get_tensor_by_name("net/features:0")
        self.image_shape = self.input_var.get_shape().as_list()[1:]
        self.feature_dim = self.output_var.get_shape().as_list()[-1]

    def encode(self, image, boxes, batch_size):
        person_images = []
        for bbox in boxes:
            person_image = extract_person(image, bbox, self.image_shape[:2])
            if person_image is None:
                person_image = np.random.uniform(0., 255., self.image_shape).astype(np.uint8)
            person_images.append(person_image)
        person_images = np.asarray(person_images)
        features = np.zeros((person_images.shape[0], self.feature_dim), np.float32)
        for i in range(0, features.shape[0], batch_size):
            features[i:i + batch_size] = self.sess.run(self.output_var,
                                                       feed_dict={self.input_var: person_images[i:i + batch_size]})
        return features
