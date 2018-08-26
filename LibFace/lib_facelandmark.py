import os
import numpy as np
import dlib
from skimage.transform import estimate_transform, warp



class FaceLandmarkDetector:

    # Member variables:
    # self.landmark_detector
    # self.resolution_inp
    # self.resolution_op

    def __init__(self, prefix = '.'):
        detector_path = os.path.join(prefix, 'model/shape_predictor_68_face_landmarks.dat')
        self.flm_detector = dlib.shape_predictor(detector_path)


    def detect(self, image, det_bbox = None):
        #   det_bbox.rect (ltrb)
        #   det_bbox.confidence

        if det_bbox is None:
            [h, w, _] = image.shape
            det_bbox = dlib.rectangle(0,0,w,h)
            # TODO: implement bounding box

        # if image is single channel
        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        # image should be in RGB order
        shape = self.flm_detector(image, det_bbox)
        
        return shape
            # shape.part(0), shape.part(1) ...
    
    
    
