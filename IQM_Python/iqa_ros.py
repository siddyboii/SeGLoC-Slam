#!/usr/bin/env python
import numpy as np
import cv2
import warnings
import math
import iqm as image_quality

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64

warnings.filterwarnings("ignore")


class IQA(Node):
    def __init__(self):
        super().__init__("IQA")
        self.get_logger().info('IQA node is started')
        self.topic_name = "/camera_front/color/image_raw"
        self.score_topic = "/image_quality_score"
        self.image_sub = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10)
        self.image_sub
        self.br = CvBridge()
        self.iqa_score_pub = self.create_publisher(
            Float64, self.score_topic, 10)
        self.iqa = image_quality.IQM()
        self.w1 = self.w2 = self.w3 = self.w4 = 0.5
        self.w5 = 0.2

    def image_callback(self, msg):
        current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        brightness, contrast = self.iqa.image_brightness(image_gray, None)
        entropy = self.iqa.image_entropy(image_gray)
        Lgradient, gradient_image = self.iqa.gradient_mapping(I=image_gray)
        awgn = self.iqa.estimate_noise(image_gray)
        _, NoiseImage, noise = self.iqa.estimate(image_gray)
        Noise = (noise*0.5+awgn*0.5)/20
        iqm_metric = self.w1*brightness+self.w2*contrast + \
            self.w3*entropy+self.w4*Lgradient-self.w5*Noise

        # NORMALIZE using sigmoid function to [0, 1] range
        # Sigmoid: f(x) = 1 / (1 + exp(-k*(x - center)))
        # k=3.0 controls steepness, center=0.5 maps median score to 0.5
        iqm_normalized = 1.0 / (1.0 + math.exp(-3.0 * (iqm_metric - 0.5)))

        # self.get_logger().info(
        #     f"IQA score: raw={iqm_metric:.4f}, normalized={iqm_normalized:.4f} "
        #     f"[brightness={brightness:.3f}, contrast={contrast:.3f}, entropy={entropy:.3f}, "
        #     f"gradient={Lgradient:.3f}, noise={Noise:.3f}]")
        score_msg = Float64()
        score_msg.data = float(iqm_normalized)
        self.iqa_score_pub.publish(score_msg)
        self.get_logger().debug('publishing the normalized score on topic /image_quality_score')

        return iqm_metric


if __name__ == "__main__":
    rclpy.init(args=None)
    iqa = IQA()
    rclpy.spin(iqa)
    rclpy.shutdown()
