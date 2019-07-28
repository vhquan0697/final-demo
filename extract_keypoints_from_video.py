# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
#params["model_folder"] = "../../../models/"
params["model_folder"] = "/media/vhquan/APCS - Study/Thesis/openpose/models"
params['keypoint_scale'] = 4
#params['write_json'] = True
params['number_people_max'] = 2
# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    #imagePaths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()

    #capture from camera in realtime
    #stream = cv2.VideoCapture(0)

    #capture from video
    stream = cv2.VideoCapture('/media/vhquan/APCS - Study/Thesis/Skeleton dataset/NTU RGB+D Dataset/nturgb+d_rgb_s018/nturgb+d_rgb/S018C003P008R001A106_rgb.avi')
    # Process and display images
    
    #total frames in video
    length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    f = open("mydata.skeleton", "w+")
    f.write(str(length) + '\n') # number of frames in video
    while True:
        ret,img = stream.read()
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])

        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #print(type(datum.poseKeypoints))
        #datum.poseKeypoints = np.reshape(datum.poseKeypoints, (2,75))
        #print(datum.poseKeypoints.tolist())
        
        f.write(str(len(datum.poseKeypoints)) + '\n') # number of people
        
        for i in range(len(datum.poseKeypoints)):
            if i == 0: 
                f.write('10543973049575027\n') #random tracking id of the skeleton
            else:
                f.write('72057594037931846\n') #random tracking id of the skeleton
            f.write(str(len(datum.poseKeypoints[i])) + '\n')
            #print(str(datum.poseKeypoints[i][0]))
            body_joint_1 = str(datum.poseKeypoints[i][8].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_2 = np.add(datum.poseKeypoints[i][1], datum.poseKeypoints[i][8] ) #not done
            body_joint_2 = body_joint_2 / 2
            body_joint_2 = str(body_joint_2).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_3 = str(datum.poseKeypoints[i][1].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_4 = str(datum.poseKeypoints[i][0].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_5 = str(datum.poseKeypoints[i][5].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_6 = str(datum.poseKeypoints[i][6].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_7 = str(datum.poseKeypoints[i][7].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_8 = str(datum.poseKeypoints[i][7].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_9 = str(datum.poseKeypoints[i][2].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_10 = str(datum.poseKeypoints[i][3].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_11 = str(datum.poseKeypoints[i][4].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_12 = str(datum.poseKeypoints[i][4].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_13 = str(datum.poseKeypoints[i][12].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_14 = str(datum.poseKeypoints[i][13].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_15 = str(datum.poseKeypoints[i][14].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_16 = str(datum.poseKeypoints[i][19].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_17 = str(datum.poseKeypoints[i][9].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_18 = str(datum.poseKeypoints[i][10].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_19 = str(datum.poseKeypoints[i][11].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_20 = str(datum.poseKeypoints[i][22].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_21 = str(datum.poseKeypoints[i][1].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_22 = str(datum.poseKeypoints[i][7].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_23 = str(datum.poseKeypoints[i][7].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_24 = str(datum.poseKeypoints[i][4].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            body_joint_25 = str(datum.poseKeypoints[i][4].tolist()).strip('[').strip(']').replace(',', '') + ' 0 0 0 0 0 0 0 0'
            f.write(body_joint_1 + '\n')
            f.write(body_joint_2 + '\n')
            f.write(body_joint_3 + '\n')
            f.write(body_joint_4 + '\n')
            f.write(body_joint_5 + '\n')
            f.write(body_joint_6 + '\n')
            f.write(body_joint_7 + '\n')
            f.write(body_joint_8 + '\n')
            f.write(body_joint_9 + '\n')
            f.write(body_joint_10 + '\n')
            f.write(body_joint_11 + '\n')
            f.write(body_joint_12 + '\n')
            f.write(body_joint_13 + '\n')
            f.write(body_joint_14 + '\n')
            f.write(body_joint_15 + '\n')
            f.write(body_joint_16 + '\n')
            f.write(body_joint_17 + '\n')
            f.write(body_joint_18 + '\n')
            f.write(body_joint_19 + '\n')
            f.write(body_joint_20 + '\n')
            f.write(body_joint_21 + '\n')
            f.write(body_joint_22 + '\n')
            f.write(body_joint_23 + '\n')
            f.write(body_joint_24 + '\n')
            f.write(body_joint_25 + '\n')
        
        '''if not args[0].no_display:
            cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break'''
    f.close()
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    # print(e)
    #print("error")
    sys.exit(-1)
