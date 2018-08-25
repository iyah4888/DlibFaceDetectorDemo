import os
import numpy as np

from skimage.transform import estimate_transform, warp


class FaceDetector:
    # Member variables:
    # self.face_detector
    # self.resolution_inp
    # self.resolution_op

    def __init__(self, prefix = '.'):
        import dlib

        self.resolution_inp = 256
        self.resolution_op = 256
        
        detector_path = os.path.join(prefix, 'model/mmod_human_face_detector.dat')
        self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)


    def detect(self, image, re_scale = 1.):
        # if image is single channel
        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        # image should be in RGB order
        dets = self.face_detector(image, 1)
        #   dets[i].rect (ltrb)
        #   dets[i].confidence
        
        if len(dets) == 0:
            print('warning: no detected face')
            return None, None, None

        # recompute the bounding box to have a square box
        d = dets[0].rect
        # pdb.set_trace()
        left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
        old_size = (right - left + bottom - top)/2
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size*re_scale)

        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        crop_tform = estimate_transform('similarity', src_pts, DST_PTS)

        return center, size, crop_tform
        #   2d center point, width(height)

    def crop(self, image, tform):
        if tform is None:
            return None

        image = image/255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        cropped_image = (cropped_image*255.).astype(np.uint8)
        return cropped_image

    def detect_and_crop(self, image, re_scale = 1.):
        pos_center, pos_size, crop_transformer = self.detect(image, re_scale) # use dlib to detect face
        cropped_image = self.crop(image, crop_transformer)
        return cropped_image
        