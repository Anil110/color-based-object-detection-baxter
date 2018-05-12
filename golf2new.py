
import rospy
import roslib

import cv2.cv as cv
import cv2
import cv_bridge

import numpy
import math
import os
import sys
import string
import time
import random
import tf
from sensor_msgs.msg import Image
import baxter_interface
from moveit_commander import conversions
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
import std_srvs.srv
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

# initialise ros node
rospy.init_node("pick_and_place", anonymous = True)

# directory used to save analysis images
image_directory = os.getenv("HOME") + "/Golf/"

# locate class
class locate():
    def __init__(self, arm, distance):
        global image_directory
        # arm ("left" or "right")
        self.limb           = arm
        self.limb_interface = baxter_interface.Limb(self.limb)

        if arm == "left":
            self.other_limb = "right"
        else:
            self.other_limb = "left"

        self.other_limb_interface = baxter_interface.Limb(self.other_limb)

        # gripper ("left" or "right")
        self.gripper = baxter_interface.Gripper(arm)

        # image directory
        self.image_dir = image_directory

        # flag to control saving of analysis images
        self.save_images = True

        # required position accuracy in metres
        self.ball_tolerance = 0.005
        self.tray_tolerance = 0.05

        # number of balls found
        self.balls_found = 0

        # start positions
        self.ball_tray_x = 0.50                        # x     = front back
        self.ball_tray_y = 0.30                        # y     = left right
        self.ball_tray_z = 0.025  #0.15                      # z     = up down
        # z distance adjusted for v1.2 endpoint of cuff moved to gripper endpoint
        self.golf_ball_x = 0.50                        # x     = front back
        self.golf_ball_y = 0.00                        # y     = left right
        self.golf_ball_z = 0.13   #0.15                # z     = up down
        # z distance kept the same to view all golf balls
        self.roll        = -1.0 * math.pi              # roll  = horizontal
        self.pitch       = 0.0 * math.pi               # pitch = vertical
        self.yaw         = 0.0 * math.pi               # yaw   = rotation

        self.pose = [self.golf_ball_x, self.golf_ball_y, self.golf_ball_z,     
                     self.roll, self.pitch, self.yaw]

        # camera parameters (NB. other parameters in open_camera)
        self.cam_calib    = 0.0025                     # meters per pixel at 1 meter
        self.cam_x_offset = 0.06  #0.015 #0.045       # camera gripper offset
        self.cam_y_offset = -0.022   #-0.01
        self.width        = 960                        # Camera resolution
        self.height       = 600

        # Hough circle accumulator threshold and minimum radius.
        self.hough_accumulator = 35
        self.hough_min_radius  = 15
        self.hough_max_radius  = 35

        # threshold image
	self.threshold = numpy.empty((self.height, self.width), numpy.uint8)

        # minimum ball tray area
        self.min_area = 20000

        # callback image
	self.cv_image = numpy.empty((self.height, self.width, 3), numpy.uint8)

        # colours
        self.white = (255, 255, 255)
        self.red = (0, 0, 255)
        self.blue = (255, 0, 0)
        self.green = (0, 255, 0)
        self.black = 0

	# font parameters
	self.fontFace = cv2.FONT_HERSHEY_TRIPLEX
	self.fontScale = 3
	self.fontScale2 = 1
	self.thickness = 7
	self.position = (30, 60)

        # ball tray corners in Baxter's screen coordinates
        self.tray_corners_in_pixels = [(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)]

        # ball tray corners in Baxter coordinates
        self.ball_tray_corner = [(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)]

        # ball tray places in Baxter coordinates
        self.ball_tray_place = [(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),
                                (0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),
                                (0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)]

        # Enable the actuators
        baxter_interface.RobotEnable().enable()

        # set speed as a ratio of maximum speed
        self.limb_interface.set_joint_position_speed(0.5)
        self.other_limb_interface.set_joint_position_speed(0.5)

        # create image publisher to head monitor
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=2)

        # calibrate the gripper
        self.gripper.calibrate()

        # display the start splash screen
        self.splash_screen("Visual Servoing", "Pick and Place")


	#### IMPORTANT:  Only two cameras can be open at a time at standard resolutions, 
	# due to bandwidth limitations. The hand cameras are opened by default at boot-time.
	# open required camera
        self.open_camera(self.limb, self.width, self.height)

        # subscribe to required camera
        self.subscribe_to_camera(self.limb)

        # distance of arm to table and ball tray
        self.distance      = distance
        self.tray_distance = distance - 0.075 

        # move other arm out of harms way
        if arm == "left":
            self.baxter_ik_move("right", (0.25, -0.50, 0.2, math.pi, 0.0, 0.0))
        else:
            self.baxter_ik_move("left", (0.25, 0.50, 0.2, math.pi, 0.0, 0.0))


    ### Important Note: Default behavior on Baxter startup is for both of the 
    # hand cameras to be in operation at a resolution of 320x200 at a 
    # framerate of 25 fps.
    # open a camera and set camera parameters
    def open_camera(self, camera, x_res, y_res):

        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        else:
            sys.exit("ERROR - open_camera - Invalid camera")

        # set camera parameters
        cam.resolution          = (int(x_res), int(y_res))
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # open camera
        cam.open()


    # convert Baxter point to image pixel
    def baxter_to_pixel(self, pt, dist):
        x = (self.width / 2)                                                         \
          + int((pt[1] - (self.pose[1] + self.cam_y_offset)) / (self.cam_calib * dist))
        y = (self.height / 2)                                                        \
          + int((pt[0] - (self.pose[0] + self.cam_x_offset)) / (self.cam_calib * dist))

        return (x, y)

    # convert image pixel to Baxter point
    def pixel_to_baxter(self, px, dist):
        x = ((px[1] - (self.height / 2)) * self.cam_calib * dist)                \
          + self.pose[0] + self.cam_x_offset
        y = ((px[0] - (self.width / 2)) * self.cam_calib * dist)                 \
          + self.pose[1] + self.cam_y_offset
        return (x, y)

    # Remove artifacts and find largest object
    def look_for_ball_tray(self, threshold):
        # width, height = cv.GetSize(threshold)
	height, width = threshold.shape               # returns rows, columns
	areaArray = []
	centre = []

	contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for i, c in enumerate(contours):
	    area = cv2.contourArea(c)
	    areaArray.append(area)

	#first sort the array by area
	sortedData = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

	#find the largest contour
	largestContour = sortedData[0][1]
	max_area = sortedData[0][0]

	# find center of area
	moments = cv2.moments(largestContour)
	centre = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

        if max_area > 0:                                # if tray found
 	    cv2.circle(threshold, (centre), 9, 0, -1)   # black(0) for treshImage,
							# white(255) for cannyImage 
							# Negative thickness means that a
							# filled circle is to be drawn.
	# find tray corners
	rect = cv2.minAreaRect(largestContour)
	self.tray_corners_in_pixels = cv2.cv.BoxPoints(rect)
	self.tray_corners_in_pixels = numpy.int0(self.tray_corners_in_pixels)

        # display the modified threshold
        cv2.imshow("Modified Threshold", threshold)

        if self.save_images:
            # save threshold image
            file_name = self.image_dir + "egg_tray_threshold.jpg"
            cv2.imwrite(file_name, self.threshold)

        # 3ms wait
        cv2.waitKey(3)

        return centre                             # return centre of tray


    # camera call back function
    def camera_callback(self, data, camera_name):
        # Convert image from a ROS image message to a CV image
        try:
            self.cv_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
        except cv_bridge.CvBridgeError, e:
            print e

        # 3ms wait
        cv2.waitKey(3)

    # left camera call back function
    def left_camera_callback(self, data):
        self.camera_callback(data, "Left Hand Camera")

    # right camera call back function
    def right_camera_callback(self, data):
        self.camera_callback(data, "Right Hand Camera")

    # create subscriber to the required camera
    def subscribe_to_camera(self, camera):
        if camera == "left":
            callback = self.left_camera_callback
            camera_str = "/cameras/left_hand_camera/image"
        elif camera == "right":
            callback = self.right_camera_callback
            camera_str = "/cameras/right_hand_camera/image"
        else:
            sys.exit("ERROR - subscribe_to_camera - Invalid camera")

        camera_sub = rospy.Subscriber(camera_str, Image, callback)

    # find next object of interest
    def find_next_golf_ball(self, ball_data, iteration):
        # if only one object then object found
        if len(ball_data) == 1:
            return ball_data[0]

        # sort objects right to left
        od = []
        for i in range(len(ball_data)):
            od.append(ball_data[i])

        od.sort()

        # if one ball is significantly to the right of the others
        if od[1][0] - od[0][0] > 30:       # if ball significantly to right of the others
            return od[0]                   # return right most ball
        elif od[1][1] < od[0][1]:          # if right most ball below second ball
            return od[0]                   # return lower ball
        else:                              # if second ball below right most ball
            return od[1]                   # return lower ball

    # find gripper angle to avoid nearest neighbour
    def find_gripper_angle(self, next_ball, ball_data):
        # if only one ball any angle will do
        if len(ball_data) == 1:
            return self.yaw

        # find nearest neighbour
        neighbour = (0, 0)
        min_d2    = float(self.width * self.width + self.height * self.height)

        for i in range(len(ball_data)):
            if ball_data[i][0] != next_ball[0] or ball_data[i][1] != next_ball[1]:
                dx = float(ball_data[i][0]) - float(next_ball[0])   # NB x and y are ushort
                dy = float(ball_data[i][1]) - float(next_ball[1])   # float avoids error
                d2 = (dx * dx) + (dy * dy)
                if d2 < min_d2:
                    neighbour = ball_data[i]
                    min_d2    = d2

        # find best angle to avoid hitting neighbour
        dx = float(next_ball[0]) - float(neighbour[0])
        dy = float(next_ball[1]) - float(neighbour[1])
        if abs(dx) < 1.0:
            angle = - (math.pi / 2.0)             # avoid divide by zero
        else:
            angle = math.atan(dy / dx)            # angle in radians between -pi and pi
        angle = angle + (math.pi / 2.0)           # rotate pi / 2 radians
        if angle > math.pi / 2.0:                 # ensure angle between -pi and pi
            angle = angle - math.pi

        return - angle                            # return best angle to grip golf ball

    # if ball near any of the ball tray places
    def is_near_ball_tray(self, ball):
        for i in self.ball_tray_place:
            d2 = ((i[0] - ball[0]) * (i[0] - ball[0]))           \
               + ((i[1] - ball[1]) * (i[1] - ball[1]))
            if d2 < 0.0004:
               return True

        return False

    # Use Hough circles to find ball centres (Only works with round objects)
    def hough_it(self, n_ball, iteration):
        # create gray scale image of balls
	gray_array = numpy.empty((self.height, self.width), numpy.uint8)
        gray_array = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
	hsv_array = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        file_name = self.image_dir + "hsv_array.jpg"
        cv2.imwrite(file_name, hsv_array)
	#gray_array = cv2.medianBlur(gray_array,5)    # try this
	low_h1  = 0
	high_h1 = 40
        low_h2 = 140
        high_h2 = 190
        low_s  = 70
        high_s = 255
        low_v  = 80
        high_v = 255

        thresholded1 = cv2.inRange(hsv_array, numpy.array([low_h1, low_s, low_v]), numpy.array([high_h1, high_s, high_v]))
        thresholded2 = cv2.inRange(hsv_array, numpy.array([low_h2, low_s, low_v]), numpy.array([high_h2, high_s, high_v]))
        thresholded = cv2.addWeighted(thresholded1, 1.0, thresholded2, 1.0, 0.0)
        file_name = self.image_dir + "new.jpg"
        cv2.imwrite(file_name, thresholded)

        #Morphological opening (remove small objects from the foreground)
        thresholded = cv2.erode(thresholded, numpy.ones((2,2), numpy.uint8), iterations=2)
        thresholded = cv2.dilate(thresholded, numpy.ones((2,2), numpy.uint8), iterations=2)

        #Morphological closing (fill small holes in the foreground)
        thresholded = cv2.dilate(thresholded, numpy.ones((15,15), numpy.uint8), iterations=2)
        thresholded = cv2.erode(thresholded, numpy.ones((15,15), numpy.uint8), iterations=2)
	mask = thresholded
		
	gray_array = cv2.bitwise_and(gray_array,thresholded,mask = mask)
		

        if self.save_images:
            # save image of Hough circles on raw image
            file_name = self.image_dir + "gray_array.jpg"
            cv2.imwrite(file_name, gray_array)
	

        # find Hough circles
	circles = cv2.HoughCircles(gray_array,cv.CV_HOUGH_GRADIENT,1,20,param1=50,
           param2=self.hough_accumulator, minRadius=self.hough_min_radius,maxRadius=self.hough_max_radius) #-- added
                            #param1=50,param2=30,minRadius=20,maxRadius=35)

        # Check for at least one ball found
        if circles is None:
            # display no balls found message on head display
            self.splash_screen("No balls", "found")
            # no point in continuing so exit with error message
            sys.exit("ERROR - hough_it - No golf balls found")

        circles = numpy.uint16(numpy.around(circles))

        ball_data = {}
        n_balls   = 0

        # copy the image
	circle_image = self.cv_image.copy() 

        # check if golf ball is in ball tray
        for i in circles[0,:]:
            # convert to baxter coordinates
            ball = self.pixel_to_baxter((i[0], i[1]), self.tray_distance)

            if self.is_near_ball_tray(ball):
                # draw the outer circle in red
                cv2.circle(circle_image, (i[0], i[1]), i[2], (0, 0, 255), 2)
                # draw the center of the circle in red
                cv2.circle(circle_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            elif i[1] > 800:
                # draw the outer circle in red
                cv2.circle(circle_image, (i[0], i[1]), i[2], (0, 0, 255), 2)
                # draw the center of the circle in red
                cv2.circle(circle_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            else:
                # draw the outer circle in green
                cv2.circle(circle_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle in green
                cv2.circle(circle_image, (i[0], i[1]), 2, (0, 255, 0), 3)

                ball_data[n_balls]  = (i[0], i[1], i[2])
                n_balls            += 1

        cv2.imshow("Hough Circle", circle_image)

	# temporary
        if self.save_images:
            # save image of Hough circles on raw image
            file_name = self.image_dir             \
		+ "all_hough_circles_" + str(n_ball) + "_" + str(iteration) + ".jpg"
            cv2.imwrite(file_name, circle_image)

        # 3ms wait
        cv2.waitKey(3)

        # display image on head monitor
        s = "Searching for golf balls"
        cv2.putText(circle_image, s, self.position, self.fontFace, self.fontScale2, self.white)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(circle_image, encoding="bgr8")
        self.pub.publish(msg)

        if self.save_images:
            # save image of Hough circles on raw image
            file_name = self.image_dir                                                 \
                      + "hough_circle_" + str(n_ball) + "_" + str(iteration) + ".jpg"
            cv2.imwrite(file_name, circle_image)

        # Check for at least one ball found
        if n_balls == 0:                    # no balls found
            # display no balls found message on head display
            self.splash_screen("No balls", "found")
            # less than 12 balls found, no point in continuing, exit with error message
            sys.exit("ERROR - hough_it - No golf balls found")

        # select next ball and find it's position
        next_ball = self.find_next_golf_ball(ball_data, iteration)

        # find best gripper angle to avoid touching neighbouring ball
        angle = self.find_gripper_angle(next_ball, ball_data)

        # return next golf ball position and pickup angle
        return next_ball, angle

    # move a limb
    def baxter_ik_move(self, limb, rpy_pose):
        quaternion_pose = conversions.list_to_pose_stamped(rpy_pose, "base")

        node = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        ik_service = rospy.ServiceProxy(node, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")

        ik_request.pose_stamp.append(quaternion_pose)
        try:
            rospy.wait_for_service(node, 5.0)
            ik_response = ik_service(ik_request)

        except (rospy.ServiceException, rospy.ROSException), error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
            sys.exit("ERROR - baxter_ik_move - Failed to append pose")

        if ik_response.isValid[0]:
            print("PASS: Valid joint configuration found")
            # convert response to joint position control dictionary
            limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
            # move limb
            if self.limb == limb:
                self.limb_interface.move_to_joint_positions(limb_joints)
            else:
                self.other_limb_interface.move_to_joint_positions(limb_joints)
        else:
            # display invalid move message on head display
            self.splash_screen("Invalid", "move")
            # little point in continuing so exit with error message
            sys.exit("ERROR - baxter_ik_move - No valid joint configuration found")

        if self.limb == limb:               # if working arm
            quaternion_pose = self.limb_interface.endpoint_pose()
            position        = quaternion_pose['position']

            # if working arm remember actual (x,y) position achieved
            self.pose = [position[0], position[1],                                \
                         self.pose[2], self.pose[3], self.pose[4], self.pose[5]]

    # update pose in x and y direction
    def update_pose(self, dx, dy):
        x = self.pose[0] + dx
        y = self.pose[1] + dy
        pose = [x, y, self.pose[2], self.roll, self.pitch, self.yaw]
        self.baxter_ik_move(self.limb, pose)

    # used to place camera over the ball tray
    def ball_tray_iterate(self, iteration, centre):
        # print iteration number
        print "Egg Tray Iteration ", iteration

        # find displacement of object from centre of image
        pixel_dx    = (self.width / 2) - centre[0]
        pixel_dy    = (self.height / 2) - centre[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.tray_distance)

        x_offset = - pixel_dy * self.cam_calib * self.tray_distance
        y_offset = - pixel_dx * self.cam_calib * self.tray_distance

        # if error in current position too big
        if error > self.tray_tolerance:
            # correct pose
            self.update_pose(x_offset, y_offset)
            # find new centre
            centre = self.threshold_it(iteration)

            # find displacement of object from centre of image
            pixel_dx    = (self.width / 2) - centre[0]
            pixel_dy    = (self.height / 2) - centre[1]
            pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
            error       = float(pixel_error * self.cam_calib * self.tray_distance)

        return centre, error

    # randomly adjust a pose to dither arm position
    # used to prevent stalemate when looking for ball tray
    def dither(self):
        x = self.ball_tray_x
        y = self.ball_tray_y + (random.random() / 10.0)
        pose = (x, y, self.ball_tray_z, self.roll, self.pitch, self.yaw)

        return pose

    # find the ball tray
    def threshold_it(self, iteration):
        if self.save_images:
            # save raw image of ball tray
            file_name = self.image_dir + "ball_tray_" + str(iteration) + ".jpg"
            cv2.imwrite(file_name, self.cv_image)

        # create an empty image variable, the same dimensions as our camera feed.
	gray = numpy.empty((self.height, self.width), numpy.uint8)

        # convert the image to a grayscale image
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        # display image on head monitor
	s = "Looking for ball tray"
        cv2.putText(self.cv_image, s, self.position, self.fontFace, self.fontScale2,   \
		self.white)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
        self.pub.publish(msg)

	# create a threshold image of the greyscale image
	ret, self.threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # display the threshold transformation
	cv2.imshow("Threshold Image", self.threshold)

        if self.save_images:
            # save threshold image of ball tray
            file_name = self.image_dir + "threshold_tray_" + str(iteration) + ".jpg"
            cv2.imwrite(file_name, self.threshold)

        # 3ms wait
        cv2.waitKey(3)

        ball_tray_centre = self.look_for_ball_tray(self.threshold)

        while ball_tray_centre[0] == 0:
            if random.random() > 0.6:
                self.baxter_ik_move(self.limb, self.dither())

            ball_tray_centre = self.threshold_it(iteration)

        return ball_tray_centre

    # find places for golf balls
    def find_places(self):

	# create temporary array for tray corners
	c = self.tray_corners_in_pixels

        # find long side of ball tray
	l1_sq = numpy.sqrt(numpy.sum((c[0] - c[1])**2))
	l2_sq = numpy.sqrt(numpy.sum((c[1] - c[2])**2))

	# create a new list cc that has the coordinates of tray_corners_in_pixels
	cc = []
	cc.extend(self.tray_corners_in_pixels)

	# if side c[1] to c[2] is longer than side c[0] to c[1]
	# then reorder the tray corners to have the coordinates 
	# of the long side first
        if l1_sq < l2_sq:                     
	    item = cc[0]
	    del cc[0]
	    cc. append(item)

        # ball tray corners in baxter coordinates
        for i in range(4):
            self.ball_tray_corner[i] = self.pixel_to_baxter(cc[i], self.tray_distance)

        # ball tray places in pixel coordinates
        ref_x = cc[0][0]
        ref_y = cc[0][1]
        dl_x  = (cc[1][0] - cc[0][0]) / 8
        dl_y  = (cc[1][1] - cc[0][1]) / 8
        ds_x  = (cc[2][0] - cc[1][0]) / 6
        ds_y  = (cc[2][1] - cc[1][1]) / 6

        p     = {}
        p[0]  = (ref_x + (3 * dl_x) + (3 * ds_x), ref_y + (3 * dl_y) + (3 * ds_y))
        p[1]  = (ref_x + (5 * dl_x) + (3 * ds_x), ref_y + (5 * dl_y) + (3 * ds_y))
        p[2]  = (ref_x + (3 * dl_x) + (1 * ds_x), ref_y + (3 * dl_y) + (1 * ds_y))
        p[3]  = (ref_x + (5 * dl_x) + (1 * ds_x), ref_y + (5 * dl_y) + (1 * ds_y))
        p[4]  = (ref_x + (3 * dl_x) + (5 * ds_x), ref_y + (3 * dl_y) + (5 * ds_y))
        p[5]  = (ref_x + (5 * dl_x) + (5 * ds_x), ref_y + (5 * dl_y) + (5 * ds_y))
        p[6]  = (ref_x + (1 * dl_x) + (3 * ds_x), ref_y + (1 * dl_y) + (3 * ds_y))
        p[7]  = (ref_x + (7 * dl_x) + (3 * ds_x), ref_y + (7 * dl_y) + (3 * ds_y))
        p[8]  = (ref_x + (1 * dl_x) + (1 * ds_x), ref_y + (1 * dl_y) + (1 * ds_y))
        p[9]  = (ref_x + (7 * dl_x) + (1 * ds_x), ref_y + (7 * dl_y) + (1 * ds_y))
        p[10] = (ref_x + (1 * dl_x) + (5 * ds_x), ref_y + (1 * dl_y) + (5 * ds_y))
        p[11] = (ref_x + (7 * dl_x) + (5 * ds_x), ref_y + (7 * dl_y) + (5 * ds_y))

        for i in range(12):
            # mark position of ball tray places
            cv2.circle(self.cv_image, (int(p[i][0]), int(p[i][1])), 5, (0, 250, 0), -1)

            # ball tray places in baxter coordinates
            self.ball_tray_place[i] = self.pixel_to_baxter(p[i], self.tray_distance)

        # display the ball tray places
        cv2.imshow("Egg tray", self.cv_image)

        if self.save_images:
            # save ball tray image with overlay of ball tray and ball positions
            file_name = self.image_dir + "ball_tray.jpg"
            cv2.imwrite(file_name, self.cv_image)

        # 3ms wait
        cv2.waitKey(3)

     # find the ball tray
    def find_ball_tray(self):
        ok = False
        while not ok:
            ball_tray_centre = self.threshold_it(0)

            error     = 2 * self.tray_tolerance
            iteration = 1

            # iterate until arm over centre of tray
            while error > self.tray_tolerance:
                ball_tray_centre, error = self.ball_tray_iterate(iteration,   \
                                          ball_tray_centre)
                iteration              += 1

	    ok = True;

	# draw bounding rectangle for ball tray
	cv2.drawContours(self.cv_image, [self.tray_corners_in_pixels], 0, (0, 255, 0), 2)

        # display tray image
        cv2.imshow("Ball Tray", self.cv_image)

        self.find_places()

    # used to place camera over golf ball
    def golf_ball_iterate(self, n_ball, iteration, ball_data):
        # print iteration number
        print "GOLF BALL", n_ball, "ITERATION ", iteration

        # find displacement of ball from centre of image
        pixel_dx    = (self.width / 2) - ball_data[0]
        pixel_dy    = (self.height / 2) - ball_data[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.tray_distance)

        x_offset = - pixel_dy * self.cam_calib * self.tray_distance
        y_offset = - pixel_dx * self.cam_calib * self.tray_distance

        # update pose and find new ball data
        self.update_pose(x_offset, y_offset)
        ball_data, angle = self.hough_it(n_ball, iteration)

        # find displacement of ball from centre of image
        pixel_dx    = (self.width / 2) - ball_data[0]
        pixel_dy    = (self.height / 2) - ball_data[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.tray_distance)

        return ball_data, angle, error

    # print all 6 arm coordinates (only required for programme development)
    def print_arm_pose(self):
        #return
        pi = math.pi

        quaternion_pose = self.limb_interface.endpoint_pose()
        position        = quaternion_pose['position']
        quaternion      = quaternion_pose['orientation']
        euler           = tf.transformations.euler_from_quaternion(quaternion)

        print
        print "             %s" % self.limb
        print 'front back = %5.4f ' % position[0]
        print 'left right = %5.4f ' % position[1]
        print 'up down    = %5.4f ' % position[2]
        #print 'roll       = %5.4f radians %5.4f degrees' %euler[0], 180.0 * euler[0] / pi
        #print 'pitch      = %5.4f radians %5.4f degrees' %euler[1], 180.0 * euler[1] / pi
        #print 'yaw        = %5.4f radians %5.4f degrees' %euler[2], 180.0 * euler[2] / pi
        return
    # find all the golf balls and place them in the ball tray
    def pick_and_place(self):
        n_ball = 0
        while True and n_ball < 12:              # assume no more than 12 golf balls
            n_ball          += 1
            iteration        = 0
            angle            = 0.0

            # use Hough circles to find balls and select one ball
            next_ball, angle = self.hough_it(n_ball, iteration)

            error     = 2 * self.ball_tolerance

            print
            print "Ball number ", n_ball
            print "==============="

            # iterate to find next golf ball
            # if hunting to and fro accept error in position
            while error > self.ball_tolerance and iteration < 10:
                iteration               += 1
                next_ball, angle, error  = self.golf_ball_iterate(n_ball, iteration, next_ball)

            s        = "Picking up golf ball"
            cv2.putText(self.cv_image, s, self.position, self.fontFace, self.fontScale2,  \
		self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            print "DROPPING BALL ANGLE =", angle * (math.pi / 180)
            if angle != self.yaw:             # if neighbouring ball
                pose = (self.pose[0],         # rotate gripper to avoid hitting neighbour
                        self.pose[1],
                        self.pose[2],
                        self.pose[3],
                        self.pose[4],
                        angle)
                self.baxter_ik_move(self.limb, pose)

                # display current image on head display
                cv2.putText(self.cv_image, s, self.position, self.fontFace, self.fontScale2, \
		    self.white)
                msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
                self.pub.publish(msg)

            # slow down to reduce scattering of neighbouring golf balls
            self.limb_interface.set_joint_position_speed(0.1)

            # move down to pick up ball
            pose = (self.pose[0] + self.cam_x_offset,
                    self.pose[1] + self.cam_y_offset,
#                    self.pose[2] + (0.112 - self.distance),
                    self.pose[2] + (0.01 - self.distance),
                    # height adjusted for gripper endpoint
                    # increasing 0.01 will cause the gripper to grip higher on the ball
                    self.pose[3],
                    self.pose[4],
                    angle)
            self.baxter_ik_move(self.limb, pose)
            self.print_arm_pose()

            # close the gripper
            self.gripper.close()

            s = "Moving to ball tray"
            cv2.putText(self.cv_image, s, self.position, self.fontFace, self.fontScale2,   \
			self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            pose = (self.pose[0],
                    self.pose[1],
                    self.pose[2] + 0.198,  
                    self.pose[3],
                    self.pose[4],
                    self.yaw)
            self.baxter_ik_move(self.limb, pose)

            # speed up again
            self.limb_interface.set_joint_position_speed(0.5)

            # display current image on head display
            cv2.putText(self.cv_image, s, self.position, self.fontFace, self.fontScale2,   \
			self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            # move down
            pose = (self.ball_tray_place[n_ball - 1][0],
                    self.ball_tray_place[n_ball - 1][1],
                    self.pose[2] - 0.19,    #  ~7.5 inches
#                    self.pose[2] - 0.127,    #  ~5 inches 
                    self.pose[3],
                    self.pose[4],
                    self.pose[5])
            self.baxter_ik_move(self.limb, pose)

            # display current image on head display
            s = "Placing golf ball in ball tray"
            cv2.putText(self.cv_image, s, self.position, self.fontFace, self.fontScale2,    \
			self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            # open the gripper
            self.gripper.open()

            # prepare to look for next ball
            pose = (self.golf_ball_x,
                    self.golf_ball_y,
                    self.golf_ball_z,
                    -1.0 * math.pi,
                    0.0 * math.pi,
                    0.0 * math.pi)
            self.baxter_ik_move(self.limb, pose)

        # display all balls found on head display
        self.splash_screen("All balls", "found")

        print "All balls found"

    # display message on head display
    def splash_screen(self, s1, s2):
        splash_array = numpy.zeros((self.height, self.width, 3), numpy.uint8)

        ((text_x, text_y), baseline) = cv2.getTextSize(s1, self.fontFace, self.fontScale, \
		self.thickness)

        org = ((self.width - text_x) / 2, (self.height / 3) + (text_y / 2))

        cv2.putText(splash_array, s1, org, self.fontFace, self.fontScale, self.green,   \
		self.thickness)

        ((text_x, text_y), baseline) = cv2.getTextSize(s2, self.fontFace, self.fontScale,  \
		self.thickness)

        org = ((self.width - text_x) / 2, ((2 * self.height) / 3) + (text_y / 2))

        cv2.putText(splash_array, s2, org, self.fontFace, self.fontScale, self.green,   \
		self.thickness)

	# publish text to x_display
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(splash_array, encoding="bgr8")
        self.pub.publish(msg)
	time.sleep(1)

# read the setup parameters from setup.dat
def get_setup():
    global image_directory
    file_name = image_directory + "setup.dat"

    try:
        f = open(file_name, "r")
    except ValueError:
        sys.exit("ERROR: golf_setup must be run before golf")

    # find limb
    s = string.split(f.readline())
    if len(s) >= 3:
        if s[2] == "left" or s[2] == "right":
            limb = s[2]
        else:
            sys.exit("ERROR: invalid limb in %s" % file_name)
    else:
        sys.exit("ERROR: missing limb in %s" % file_name)

    # find distance to table
    s = string.split(f.readline())
    if len(s) >= 3:
        try:
            distance = float(s[2])
        except ValueError:
            sys.exit("ERROR: invalid distance in %s" % file_name)
    else:
        sys.exit("ERROR: missing distance in %s" % file_name)

    return limb, distance

def main():
    # get setup parameters
    limb, distance = get_setup()
    print "limb     = ", limb
    print "distance = ", distance

    # create locate class instance
    locator = locate(limb, distance)

    raw_input("Press Enter to start: ")

    # open the gripper
    locator.gripper.open()

    # move close to the ball tray
    locator.pose = (locator.ball_tray_x,
                    locator.ball_tray_y,
                    locator.ball_tray_z,
                    locator.roll,
                    locator.pitch,
                    locator.yaw)
    locator.baxter_ik_move(locator.limb, locator.pose)

    # find the ball tray
    locator.find_ball_tray()

    # find all the golf balls and place them in the ball tray
    locator.pose = (locator.golf_ball_x,
                    locator.golf_ball_y,
                    locator.golf_ball_z,
                    locator.roll,
                    locator.pitch,
                    locator.yaw)

    locator.baxter_ik_move(locator.limb, locator.pose)

    locator.pick_and_place()

if __name__ == "__main__":
    main()

