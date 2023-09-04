#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
import cv2

# Taken from: https://github.com/daniilidis-group/EV-FlowNet
"""
Generates an RGB image where each point corresponds to flow in that direction from the center,
as visualized by flow_viz_tf.
Output: color_wheel_rgb: [1, width, height, 3]
"""
def draw_color_wheel_tf(width, height):
    color_wheel_x = tf.lin_space(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = tf.lin_space(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = tf.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_flow = tf.stack([color_wheel_X, color_wheel_Y], axis=2)
    color_wheel_flow = tf.expand_dims(color_wheel_flow, 0)
    color_wheel_rgb, flow_norm, flow_ang = flow_viz_tf(color_wheel_flow)
    return color_wheel_rgb

"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [batch_size, width, height, 2]
Output: flow_rgb: [batch_size, width, height, 3]
"""
def flow_viz_tf(flow):
    flow_norm = tf.norm(flow, axis=3)
    
    flow_ang_rad = tf.atan2(flow[:, :, :, 1], flow[:, :, :, 0])
    flow_ang = (flow_ang_rad / math.pi) / 2. + 0.5
    
    const_mat = tf.ones(tf.shape(flow_norm))
    hsv = tf.stack([flow_ang, const_mat, flow_norm], axis=3)
    flow_rgb = tf.image.hsv_to_rgb(hsv)
    return flow_rgb, flow_norm, flow_ang_rad

