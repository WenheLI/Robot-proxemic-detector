# Import the necessary libraries
from copyreg import constructor
from importlib.resources import path
import os
import rclpy                                    # Python library for ROS 2
from rclpy.node import Node                     # Handles the creation of nodes
from sensor_msgs.msg import Image               # Image is the message type
import cv2                                      # OpenCV library
from cv_bridge import CvBridge, CvBridgeError   # Package to convert between ROS and OpenCV Images
import numpy as np                              # Used to modify image data
from std_msgs.msg import String                 # Used to process ROS images
from geometry_msgs.msg import Twist             # Sends velocity commands to the robot
from gtts import gTTS

class ProxemicDetection(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """
    def __init__(self, proxemic_ranges, display=False, color=None):
        # Initiate the Node class's constructor and give it a name
        super().__init__('turtlebot_proxemic_detector_node')

        if not os.path.exists('./close.mp3'):
            myobj = gTTS(text="Hi, you are so close", lang="en", slow=False)
            myobj.save("./close.mp3")

            myobj = gTTS(text="Hi, you are in the okay distance", lang="en", slow=False)
            myobj.save("./normal.mp3")

            
        # Initialize variables
        self.DISPLAY = display
        self.color = color
        self.proxemic_ranges=proxemic_ranges

        # RGB variables
        self.rgb_image_labeled = None
        self.rgb_image = None
        self.rgb_bridge = CvBridge()

        # TASK 1: Subscribe to depth topic
        self.rgb_subscription = self.create_subscription(
            Image, "/color/preview/image", self.rgb_callback, 10
        )
        '''
        self.rgb_subscription = self.create_subscription(
            )
        '''
        
        
        # Depth Variables
        self.depth_image = None
        self.depth_bridge = CvBridge()

        # TASK 1: Subscribe to depth topic
        '''
        self.depth_subscription = self.create_subscription(
            )
        '''

        self.depth_subscription = self.create_subscription(
            Image, "/stereo/depth", self.depth_callback, 10
        )
        
        # TASK 2: Create a publisher for the /cmd_vel topic
        self.cmd_vel_publisher = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        # TASK 3: Create a publisher for the /cmd_vel topic
        self.proxemic_publisher = self.create_publisher(
            String, "/proxemic", 10
        )

        # Initialize bounding boxes
        self.bboxes = {'red':[],'green':[],'blue':[]}
    
    def rgb_callback(self, msg):
        """Convert ROS RGB sensor_msgs to opencv image
        ----------
        msg : Depth sensor_msgs image data
            A Depth image of format sensor_msgs Image with one channel measuring the distance to objects.
        Returns
        -------
        None
        """
        try:
            # TASK 1: Convert ROS Image message to OpenCV image with imgmsg_to_cv2 function which is apart of ROS brigde object self.rgb_bridge
            #self.rgb_image = self.rgb_bridge ...

            self.rgb_image = self.rgb_bridge.imgmsg_to_cv2(msg, "bgr8")
            print("rgb ", self.rgb_image.shape)

            # Task 1:Run color detection on image to generate bounding box color blobs
            # Uncomment next line to generate color blobs in self.bboxes
            # self.rgb_image_labeled, self.bboxes = self.read_process_image_color(...)
            
            # Display image
            self.rgb_image_labeled, self.bboxes = self.read_process_image_color(self.rgb_image, self.color)
            if(self.DISPLAY and self.rgb_image_labeled is not None):
                cv2.imshow("RGB Image", self.rgb_image_labeled)
                cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

        if(self.rgb_image is None):
            self.get_logger().info('Failed to load RGB image')

    def depth_callback(self, msg): 
        """Convert ROS depth sensor_msgs to opencv image
        ----------
        msg : Depth sensor_msgs image data
            A Depth image of format sensor_msgs Image with one channel measuring the distance to objects.
        Returns
        -------
        None
        """
        # Convert ROS Image message to OpenCV image
        try:
            # TASK 1: Convert ROS Image message to OpenCV image with imgmsg_to_cv2 function which is apart of ROS brigde object self.depth_bridge
            #self.depth_image = self.depth_bridge... 
            self.depth_image_1 = self.depth_bridge.imgmsg_to_cv2(msg, "passthrough")
            self.depth_image = cv2.resize(self.depth_image_1, (self.rgb_image.shape[0], self.rgb_image.shape[1]))
            # Display image
            if(self.DISPLAY and self.depth_image is not None):
                cv2.imshow("Depth Image", self.depth_image)
                cv2.imshow("Depth Raw Image", self.depth_image_1)
                cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

        if(self.depth_image is None):
            self.get_logger().info('Failed to load Depth image')

        # TASK 2: Detect the distance to objects (Uncomment next line)
        self.detection_object_distance()

    def color_detection(self, img, hsvFrame, kernal, min_width, bounding_boxes, lower_bound, upper_bound, filter, color=None):
        """Apply color filter to RGB image
        ----------
        img : RGB image data
            A RGB image of format numpy array with RGB channels.
        hsvFrame : filename of input image
            The image_filename is format a string with the path image.
        kernal : opencv image kernal 
        min_width : RGB image data
            A minmum width of bounding boxes of format int.
        bounding_boxes : dict
            Dictionary of bounding boxes for colors
        lower_bound : numpy array
            The upper bound for color is format array of numpy array with filter values.
        upper_bound : numpy array
            The upper bound for color is format array of numpy array with filter values.
        color : color to idenity in images. 
            The color is format string. Options include 'red', 'green', 'blue', or None for all colors
        Returns
        -------
        img
            Annotated image with color bounding boxes
        hsvFrame, bounding_boxes, kernal (ass as input)
        """
        # Set upper and lower bound for color detection
        mask = cv2.inRange(hsvFrame, lower_bound, upper_bound)
        mask = cv2.dilate(mask, kernal)

        # Creating contour to track  color
        contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        # Generate color blob contours and draw boxes around them on image
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                if(w < 50 or w > min_width): continue
                bounding_boxes[color].append([x, y, w, h])
                img = cv2.rectangle(img, (x, y), 
                                        (x + w, y + h), 
                                        filter, 2)
                cv2.putText(img, color+" colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            filter)  

        return img, hsvFrame, bounding_boxes, kernal

    def read_process_image_color(self, image, image_filename=None, color=None):
        """Read in image or take an image as input, run color filtering for red, green, and blue.
        Parameters
        ----------
        image : RGB image data
            A RGB image of format numpy array with RGB channels.
        image_filename : filename of input image
        The image_filename is format a string with the path image.
        color : color to idenity in images. 
            The color is format string. Options include 'red', 'green', 'blue', or None for all colors
        Returns
        -------
        image
            Image with bounding boxes that show color(s) annotated on the input image
        bounding_boxes
            List of dicts of bounding box coordinates for color blobs with fields 'red', 'green', blue
        """
        # Reading the image
        if(image_filename is None):
            img = image
        else:
            img = cv2.imread(image_filename)
        
        width, height, channels = img.shape
        min_width = width/4
        kernal = np.ones((5, 5), "uint8")
        bounding_boxes = {'red':[], 'green':[], 'blue':[]}
        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set range for red color and define mask
        if(color == 'red' or color is None):
            red_lower = np.array([136, 87, 111], np.uint8)
            red_upper = np.array([180, 255, 255], np.uint8)
            filter = (0, 0, 255)
            img, hsvFrame, bounding_boxes, kernal = self.color_detection(img, 
                                                                    hsvFrame, 
                                                                    kernal, 
                                                                    min_width, 
                                                                    bounding_boxes,  
                                                                    red_lower, 
                                                                    red_upper,
                                                                    filter, 
                                                                    color='red')

        # Set range for green color and 
        # define mask
        if(color == 'green' or color is None):
            green_lower = np.array([25, 52, 72], np.uint8)
            green_upper = np.array([102, 255, 255], np.uint8)
            filter = (0, 255, 0)
            img, hsvFrame, bounding_boxes, kernal = self.color_detection(img, 
                                                                    hsvFrame, 
                                                                    kernal, 
                                                                    min_width, 
                                                                    bounding_boxes, 
                                                                    green_lower, 
                                                                    green_upper, 
                                                                    filter,
                                                                    color='green')

        # Set range for blue color and
        # define mask
        if(color == 'blue' or color is None):
            blue_lower = np.array([94, 80, 2], np.uint8)
            blue_upper = np.array([120, 255, 255], np.uint8)
            filter = (255, 0, 0)
            img, hsvFrame, bounding_boxes, kernal = self.color_detection(img, 
                                                                    hsvFrame, 
                                                                    kernal, 
                                                                    min_width, 
                                                                    bounding_boxes, 
                                                                    blue_lower, 
                                                                    blue_upper, 
                                                                    filter,
                                                                    color='blue')

        return img, bounding_boxes
    
    def detection_object_distance(self):
        """Detects distance to objects in color blobs and alerts user of proxemic zone
        """
        # Tune the self.depth_threshold based on sensor readings
        self.close_object = False
        distance_to_object = 0
        img_patch_mean=None     # temporary variable to store mean depth in color blobs
        bbox_img_patch_mean=[]  # stores list of img_patch_mean for all color blobs

        # TASK 2: Process image data to detect nearby objects; set distance_to_object
        # Compute to average depth pixel distance to nearby objects
        colors = [self.color]
        if self.color is None:
            colors = ['red', 'green', 'blue']
        for color in colors:
            color_box = self.bboxes[color]
            for box in color_box:
                temp = self.extract_image_patch(self.depth_image, box)
                img_patch_mean = np.mean(temp)
                bbox_img_patch_mean.append(img_patch_mean)
        # TASK 2: Use min distance in average depth pixel distance to detect proximity and alert user
        if len(bbox_img_patch_mean) > 0:
            self.close_object = True
            distance_to_object = min(bbox_img_patch_mean)
            if self.proxemic_ranges['initmate_depth_threshold_min'] < distance_to_object < self.proxemic_ranges['initmate_depth_threshold_max']:
                print("Intimate Zone")
                # os.system("mpg321 ./close.mp3")
            print("Object is too close. Distance: ", distance_to_object)

        #if(len(bbox_img_patch_mean)>0):
        
        if(self.DISPLAY):
            # distance to object
            self.get_logger().info('Distance to object:' + str(distance_to_object))

            # True = close to object; False = not close to object
            self.get_logger().info('Publishing -> is robot close to object?: distance '+str(self.close_object)) 

    def extract_image_patch(self, image, bbox, patch_shape=(20,20)):
        """Extract image patch from bounding box.
        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.
        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.
        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

def main(args=None):
   
    # Initialize the rclpy library
    rclpy.init(args=args)
    print("Initializing node")
    # User set parameters
    display=True
    color=None
    proxemic_ranges = {'initmate_depth_threshold_min':10,
                        'initmate_depth_threshold_max':20,
                        'public_depth_threshold_min':50,
                        'public_depth_threshold_max':60}

    # Create the node. Color options include 'red', 'green', 'blue', or None for all colors
    proxemic_detector = ProxemicDetection(proxemic_ranges, display=display, color=color)

    # Spin the node so the callback function is called.
    rclpy.spin(proxemic_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    proxemic_detector.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()
   
if __name__ == '__main__':
  main()