import numpy as np
import cv2
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
# import pdb
from lib_facedet import FaceDetector, FaceLandmarkDetector
import dlib

def get_img_flist(image_folder):
    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    return sorted(image_path_list)


def main(args):

    # ---- init 
    image_folder = args.inputDir
    save_folder =  os.path.join(args.outputDir, 'results')

    # face detector object
    obj_facedetect = FaceDetector()
    obj_facelandmarkdet = FaceLandmarkDetector()
    win = dlib.image_window()

    # ------------- load data
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # read all the file list in a folder
    image_path_list = get_img_flist(image_folder)
    
    for i, image_path in enumerate(image_path_list):
        print(image_path)    
        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, _] = image.shape
        
            # resize image if so large
        # max_size = max(image.shape[0], image.shape[1]) 
        # if max_size > 1000:
            # image = rescale(image, 1000./max_size)

        # Detect face
        crop_image = obj_facedetect.detect_and_crop(image, 1.6)
            # this is equal to the following procedure
                # pos_center, pos_size, crop_transformer = obj_facedetect.detect(image) # use dlib to detect face
                # crop_image = obj_facedetect.crop(image, crop_transformer)
        shape = obj_facelandmarkdet.detect(crop_image, det_bbox = None)
        
        win.clear_overlay()
        for i in range(68):
            cv2.circle(crop_image, (shape.part(i).x, shape.part(i).y), 1, (255, 0, 0), -1)
        win.set_image(crop_image)
        dlib.hit_enter_to_continue()

        if args.isImage:
            print('[Saved] ' + os.path.join(save_folder, name + '.jpg'))
            imsave(os.path.join(save_folder, name + '.jpg'), crop_image) 
       
        # if args.isShow:
        #     # ---------- Plot
        #     image = image[:,:,::-1]
        #     cv2.imshow('face crop', crop_image)
        #     cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detector')

    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str, 
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,  
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval, 
                        help='whether to save input image')
    
    main(parser.parse_args())