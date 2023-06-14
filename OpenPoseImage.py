import cv2
import time
import numpy as np
import argparse
from PIL import Image 
import PIL 
import math

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--image_file", default="single.jpeg", help="Input image")

args = parser.parse_args()


MODE = "COCO"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


frame = cv2.imread(args.image_file)
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

t = time.time()
# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
        print(points)
        
    else :
        points.append(None)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)

cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)
print("Total time taken : {:.3f}".format(time.time() - t))

# Taille en cm / taille en pixel
MAC_PIXEL_TO_CM = 0.12454
BACK_CAMERA_PIXEL_TO_CM = 0.09381898455
FRONT_CAMERA_PIXEL_TO_CM = 0.1019184652

if MODE is "COCO":
    print("Body Measurments:")
    rightShoulderLength = math.dist(points[1],points[2])
    print("Right shoulder length:", rightShoulderLength, "px", "|", rightShoulderLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")
    rightBicepsLength = math.dist(points[2],points[3])
    print("Right biceps length:", rightBicepsLength,"px", "|", rightBicepsLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")
    rightForearmLength = math.dist(points[3],points[4])
    print("Right forearm length:", rightForearmLength,"px", "|", rightForearmLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")
    rightArmLength = rightShoulderLength + rightBicepsLength + rightForearmLength
    print("Right arm length:", rightArmLength, "px", "|", rightArmLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")



    leftShoulderLength = math.dist(points[1],points[5])
    print("Left shoulder length:", leftShoulderLength, "px", "|", leftShoulderLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")
    leftBicepsLength = math.dist(points[5],points[6])
    print("Left biceps length:", leftBicepsLength,"px",  "|", leftBicepsLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")
    leftForearmLength = math.dist(points[6],points[7])
    print("Left foremarm length:", leftForearmLength,"px",  "|", leftForearmLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")
    leftArmLength = leftShoulderLength + rightBicepsLength + leftForearmLength
    print("Left arm length:", leftArmLength, "px",  "|", leftArmLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")

    shoulderToShoulderLength = rightShoulderLength + leftShoulderLength
    print("Shoulder to shoulder length:", shoulderToShoulderLength, "px",  "|", shoulderToShoulderLength*FRONT_CAMERA_PIXEL_TO_CM,"cm")

    

cv2.waitKey(0)


