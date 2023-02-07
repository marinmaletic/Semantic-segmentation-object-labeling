import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from color_coding_dict import *
import time


class ImgFeature:

    def __init__(self):
        # Instantiate CvBridge
        self.bridge = CvBridge()

        self.input_img_raw_topic = "/camera/color/image_raw"
        self.input_img_topic = "/segmentation/color/image_raw"
        self.input_depth_topic = "/camera/aligned_depth_to_color/image_raw"
        self.output_labeled_raw_img_topic = "/camera/output/labeled_image_raw"
        self.output_labeled_segm_img_topic = "/camera/output/labeled_image_segmented"

        self.data = None
        self.img_raw = None
        self.n = 0
        self.i = 0
        self.past_points = {}


    def image_raw_callback(self, msg):
        self.img_raw = msg


    def imageDepthCallback(self, data):
        self.data = data


    def image_callback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2_raw = self.bridge.imgmsg_to_cv2(self.img_raw, "bgr8")
            labeled_segm, labeled_raw = self.parse(cv2_img, cv2_raw)
            out_img = self.bridge.cv2_to_imgmsg(labeled_segm, "bgr8")
            raw_img = self.bridge.cv2_to_imgmsg(labeled_raw, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            self.segm_image_pub.publish(out_img)
            self.raw_image_pub.publish(raw_img)


    def run(self):
        self.input_segm_sub = rospy.Subscriber(self.input_img_topic, Image, self.image_callback)
        self.input_raw_sub = rospy.Subscriber(self.input_img_raw_topic, Image, self.image_raw_callback)
        self.depth_sub = rospy.Subscriber(self.input_depth_topic, Image, self.imageDepthCallback)
        self.raw_image_pub = rospy.Publisher(self.output_labeled_raw_img_topic, Image, queue_size=1)
        self.segm_image_pub = rospy.Publisher(self.output_labeled_segm_img_topic, Image, queue_size=1)



    def parse(self, img, img_raw):
        start_time = time.time()
        # convert to hsv colorspace
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hz = 1  # ideja da se mijenjaju oznake svakih hz frameova, a ne svaki (vraća grešku nekad, pronaći bolje rješenje)
        self.n += 1
        self.i += 1
        if self.n == hz:
            self.past_points = {}

        for name in color_dict.keys():
            r, g, b = color_dict[name]
            h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            lower_bound = np.array([h, s, v])
            upper_bound = np.array([h, s, v])

            # find the colors within the boundaries
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # define kernel size
            kernel = np.ones((7, 7), np.uint8)
            # Remove unnecessary noise from mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Segment only the detected region
            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 0)

            if contours:
                for c in contours:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    if M["m00"] > 300:

                        if self.n == hz:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            self.past_points[name] = (cX, cY)
                        elif name in self.past_points.keys():
                            cX = self.past_points[name][0]
                            cY = self.past_points[name][1]

                        cv_depth_image = self.bridge.imgmsg_to_cv2(self.data, self.data.encoding)

                        cv2.circle(img, (cX, cY), 3, (255, 255, 255), -1)
                        # pristupanje pixelima ide obrnuto, za x i y mjesto je image[y, x]
                        cv2.putText(img, "{} ({} m)".format(name, self.get_mean_depth(cv_depth_image, cY, cX, 15)),
                                    (cX - 40, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        cv2.circle(img_raw, (cX, cY), 3, (255, 255, 255), -1)
                        cv2.putText(img_raw, "{} ({} m)".format(name, self.get_mean_depth(cv_depth_image, cY, cX, 15)),
                                    (cX - 40, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if self.n == hz:
            self.n = 0
        #print("Frame {} --- {} seconds ---".format(self.i, round(time.time() - start_time, 4)))
        return img, img_raw


    def get_mean_depth(self, depth_image, x, y, r):
        crop_img = depth_image[x-r:x+r, y-r:y+r]
        mean = round(np.average(crop_img) * (10 ** -3), 2)
        return mean

if __name__ == '__main__':
    ic = ImgFeature()
    rospy.init_node('image_feature')
    ic.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()
