import tensorflow as tf
import numpy as np
import cv2


INPUT_SIZE = 416
ANCHORS = [(10, 13), (16, 30), (33, 23),
           (30, 61), (62, 45), (59, 119),
           (116, 90), (156, 198), (373, 326)]

slim = tf.contrib.slim


def darknet53(inputs):
    net = conv_op(inputs, 32, 3, 1)
    net = conv_op(net, 64, 3, 2)
    net = darknet53_block(net, 32)
    net = conv_op(net, 128, 3, 2)
    for _ in range(2):
        net = darknet53_block(net, 64)

    net = conv_op(net, 256, 3, 2)
    for _ in range(8):
        net = darknet53_block(net, 128)
    route_1 = net

    net = conv_op(net, 512, 3, 2)
    for _ in range(8):
        net = darknet53_block(net, 256)
    route_2 = net

    net = conv_op(net, 1024, 3, 2)
    for _ in range(4):
        net = darknet53_block(net, 512)
    outputs = net

    return route_1, route_2, outputs


def conv_op(inputs, num_filters, kernel_size, strides):
    if strides > 1:
        inputs = tf.pad(inputs, [[0, 0], [kernel_size // 2, kernel_size // 2],
                                 [kernel_size // 2, kernel_size // 2], [0, 0]])
        outputs = slim.conv2d(inputs, num_filters, kernel_size, stride=strides, padding='VALID')
    else:
        outputs = slim.conv2d(inputs, num_filters, kernel_size, stride=strides, padding='SAME')
    return outputs


def darknet53_block(inputs, filters):
    net = conv_op(inputs, filters, 1, 1)
    net = conv_op(net, filters * 2, 3, 1)
    outputs = net + inputs
    return outputs


def spp_block(inputs):
    return tf.concat([slim.max_pool2d(inputs, 13, 1, 'SAME'),
                      slim.max_pool2d(inputs, 9, 1, 'SAME'),
                      slim.max_pool2d(inputs, 5, 1, 'SAME'),
                      inputs], axis=3)


def yolo_block(inputs, filters, with_spp=False):
    net = conv_op(inputs, filters, 1, 1)
    net = conv_op(net, filters * 2, 3, 1)
    net = conv_op(net, filters, 1, 1)

    if with_spp:
        net = spp_block(net)
        net = conv_op(net, filters, 1, 1)

    net = conv_op(net, filters * 2, 3, 1)
    outputs = conv_op(net, filters, 1, 1)
    return outputs


def detect_op(inputs, image_size, num_classes, anchors):
    grid_size = inputs.get_shape().as_list()[1]
    num_anchors = len(anchors)
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1,
                              normalizer_fn=None, activation_fn=None,
                              biases_initializer=tf.zeros_initializer())

    predictions = tf.reshape(predictions, [-1, num_anchors * grid_size * grid_size, 5 + num_classes])
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

    # The corresponding size of each grid cell on the input image
    stride = (image_size // grid_size, image_size // grid_size)

    # Convert grid cell coordinates to bounding box center coordinates on the input image
    grid_x = tf.range(grid_size, dtype=tf.float32)
    grid_y = tf.range(grid_size, dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
    box_centers = tf.sigmoid(box_centers)
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    # Convert zoom ratio of the anchor box to real bounding box size on the input image
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    anchors = tf.tile(anchors, [grid_size * grid_size, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    confidence = tf.sigmoid(confidence)
    classes = tf.sigmoid(classes)

    predictions = tf.concat([box_centers, box_sizes, confidence, classes], axis=-1)
    return predictions


def convert_result(predictions, score_threshold=0.6, max_output_size=40, iou_threshold=0.5):
    boxes = predictions[0][:, :4]

    # scores = confidence * classification probability
    scores = tf.expand_dims(predictions[0][:, 4], 1) * predictions[0][:, 5:]

    # Classification result and max score of each anchor box
    box_classes = tf.argmax(scores, axis=1)
    max_scores = tf.reduce_max(scores, axis=1)

    # Filter the anchor boxes by the max scores
    filter_mask = max_scores >= score_threshold
    scores = tf.boolean_mask(max_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    box_classes = tf.boolean_mask(box_classes, filter_mask)

    # Non Max Suppression (do not distinguish different classes)
    # box (x, y, w, h) -> _box (x1, y1, x2, y2)
    _boxes = tf.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                       boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
    nms_indices = tf.image.non_max_suppression(_boxes, scores, max_output_size, iou_threshold)

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    box_classes = tf.gather(box_classes, nms_indices)
    return scores, boxes, box_classes


def yolo_v3(inputs, num_classes, is_training=False, reuse=False, with_spp=False):

    image_size = inputs.get_shape().as_list()[1]

    batch_norm_params = {'decay': 0.9,
                         'epsilon': 1e-05,
                         'scale': True,
                         'is_training': is_training,
                         'fused': None}
    with tf.variable_scope('detector'):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, net = darknet53(inputs)

                with tf.variable_scope('yolo-v3'):
                    net = yolo_block(net, 512, with_spp)
                    output_1 = conv_op(net, 1024, 3, 1)
                    detect_1 = detect_op(output_1, image_size, num_classes, ANCHORS[6:9])

                    net = conv_op(net, 256, 1, 1)
                    upsample_size = route_2.get_shape().as_list()
                    net = tf.image.resize_nearest_neighbor(net, (upsample_size[1], upsample_size[2]))
                    net = tf.concat([net, route_2], axis=3)

                    net = yolo_block(net, 256)
                    output_2 = conv_op(net, 512, 3, 1)
                    detect_2 = detect_op(output_2, image_size, num_classes, ANCHORS[3:6])

                    net = conv_op(net, 128, 1, 1)
                    upsample_size = route_1.get_shape().as_list()
                    net = tf.image.resize_nearest_neighbor(net, (upsample_size[1], upsample_size[2]))
                    net = tf.concat([net, route_1], axis=3)

                    net = yolo_block(net, 128)
                    output_3 = conv_op(net, 256, 3, 1)
                    detect_3 = detect_op(output_3, image_size, num_classes, ANCHORS[0:3])

                    predictions = tf.concat([detect_1, detect_2, detect_3], axis=1)

                    if is_training:
                        return predictions
                    else:
                        return convert_result(predictions)


class Detector(object):
    def __init__(self, checkpoint_filename):
        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 3])
        self.scores, self.boxes, self.box_classes = yolo_v3(self.inputs, 80)
        tf.train.Saver().restore(self.sess, checkpoint_filename)

    def detect(self, frame):
        img_h, img_w, _ = frame.shape
        img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_in = img_rgb.reshape((1, INPUT_SIZE, INPUT_SIZE, 3)) / 255.
        scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.box_classes],
                                                   feed_dict={self.inputs: img_in})

        # Convert coordinates to the scale of the original image
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] // 2) * img_w / INPUT_SIZE
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] // 2) * img_h / INPUT_SIZE
        boxes[:, 2] = boxes[:, 2] * img_w / INPUT_SIZE
        boxes[:, 3] = boxes[:, 3] * img_h / INPUT_SIZE

        # Only keep person object
        scores = scores[np.where(box_classes == 0)]
        boxes = boxes[np.where(box_classes == 0)]
        box_classes = box_classes[np.where(box_classes == 0)]
        return scores, boxes, box_classes
