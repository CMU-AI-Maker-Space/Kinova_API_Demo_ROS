#!/usr/bin/env python3
import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf2_ros
from geometry_msgs.msg import Point

class image_converter:

    def __init__(self):
        # Subscribe to the raw color image from the camera
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.find_object)
        # Subscribe to get the camera info that will be used to estimate the point in
        self.camera_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_params)
        # Start the ROS-CV bridge
        self.bridge = CvBridge()
        # Assign a publisher to broadcast the image + tracking bounding circle to other nodes
        self.image_pub = rospy.Publisher("/tracking/image_with_tracking", Image, queue_size=1)

        # Object parameters
        self.ball_radius = 0.03
        self.ball_height = 0.05
        self.kinova_support = 0.02 # height of black base under the arm

        # Camera parameters (start as identity)
        self.camera_parameters = np.eye(3)

        # Frame transformation objects
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # define the lower and upper boundaries of the "green" ball in the HSV color space
        # Very ad hoc values... might need to be changed depending on lighting
        self.greenLower = (36, 25, 25)
        self.greenUpper = (70, 255, 255)

        # Blur parameters
        self.kernel_size = (11, 11)
        self.sigma = 0

        # Erosion/dilation param
        self.erosion_iterations = 20
        self.dilation_iterations = 20

    def quaternion_rotation_matrix(self, w, x, y, z):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = w
        q1 = x
        q2 = y
        q3 = z

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix

    def camera_params(self, camera_info: CameraInfo):
        """
        Saves the intrinsic parameters of the camera, which will then be accessed by the object tracker to pinpoint pos
        """
        self.camera_parameters = np.array(camera_info.K).reshape((3,3))

    def find_object(self, streamed_image):
        """
        This function will look for a green ball in the image
        It will publish an image with a tracked circle and the position of the object's center
        :param streamed_image: sensor.msg.Image
        """
        try:
            # Try to convert the image message from the camera to opencv format
            cv_image = self.bridge.imgmsg_to_cv2(streamed_image, "bgr8") #OpenCV uses bgr

            # Blur and convert to HSV
            blurred = cv2.GaussianBlur(cv_image, self.kernel_size, self.sigma)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Create a mask of color green
            mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
            mask = cv2.erode(mask, None, self.erosion_iterations)
            mask = cv2.dilate(mask, None, self.dilation_iterations)

            # Find the right contours
            cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(cv_image, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(cv_image, (int(x), int(y)), 5, (0, 0, 255), -1)

                    # Calculate the direction of the ray in the camera frame
                    px = x
                    py = y
                    cx = self.camera_parameters[0, 2]
                    cy = self.camera_parameters[1, 2]
                    cz = self.camera_parameters[2, 2]
                    fx = self.camera_parameters[0, 0]
                    fy = self.camera_parameters[1, 1]
                    pc0 = np.array([
                        (px - (cx/cz))*(cz/fx),
                        (py - (cy/cz))*(cz/fy),
                        1
                    ])
                    # Calculate the transformation between the camera frame to the base frame
                    # Target frame = frame to which data should be transformed
                    transformation = self.tfBuffer.lookup_transform('base_link', 'camera_color_frame', rospy.Time(0))
                    transform_translation = np.array([
                        transformation.transform.translation.x,
                        transformation.transform.translation.y,
                        transformation.transform.translation.z
                        ])
                    transform_rotation = self.quaternion_rotation_matrix(transformation.transform.rotation.w,
                                                                         transformation.transform.rotation.x,
                                                                         transformation.transform.rotation.y,
                                                                         transformation.transform.rotation.z,)
                    # Solve the camera ambiguity
                    # Monocular cameras can't pinpoint depth. This parameter will be estimated by knowing the target height in global coordinates
                    center_height = self.ball_height-self.ball_radius
                    cam_z = ((center_height-self.kinova_support-transform_translation[2])/
                             (transform_rotation[2, :]@pc0))
                    # Get the point in the camera frame
                    cam_point = cam_z*pc0
                    # Convert to global
                    global_point = transform_rotation@cam_point + transform_translation
                    # Save calculated point to parameter server
                    rospy.set_param("ball_center", global_point.tolist())

            # Publish image with tracking to correct rostopic
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        except CvBridgeError as e:
            # If there is an error converting the image to cv format or publishing it
            print(e)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

def main(args):
    rospy.init_node('object_tracker')
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
