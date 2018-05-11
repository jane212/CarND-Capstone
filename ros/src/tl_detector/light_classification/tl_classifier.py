from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
from numpy import newaxis
import tensorflow as tf
from keras.models import load_model
import rospy
import rospkg

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	r = rospkg.RosPack()
	path = r.get_path('tl_detector')
	#print(path)
        self.model = load_model(path+'/classification_model.h5')
	self.model._make_predict_function()
	self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
	img = cv2.resize(image, (400,400))
	img = img.astype(float)
	img = img/255.0
	img = img[newaxis,:,:,:]

	with self.graph.as_default():
		preds = self.model.predict(img)
		#rospy.loginfo("prob: "+str(preds))
	prediction = np.argmax(preds, axis=1)
	light_id = prediction[0]

	if light_id==1:
		return TrafficLight.RED
	return TrafficLight.UNKNOWN 
	
        return TrafficLight.UNKNOWN
