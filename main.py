import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# RCNN Dependencies
import utils
import coco
import utils
import model as modellib
import visualize

# Video Detection Dependenices
import cv2
import time
import urllib.request as urllib2

# particle filter dependencies
import particle
import particleFilter as pf

if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")


    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    ##################################################
    # Real Time Detection
    ##################################################
    plt.ion()
    size = (16, 16)
    fig, ax = plt.subplots(1, figsize = size)

    cap = cv2.VideoCapture(0)

    if (cap.isOpened() == False):
        print("Error opening video stream / file")
        exit(0)
    else:
        ret , frame0 = cap.read()
        prevFrame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    ##################################################
    # Particle Filter Initialization
    ##################################################
    numParticles = 100
    particles = [particle.Particle(0,0,0)]*numParticles
    for i in range(numParticles):
        particles[i] = particle.Particle(random.randint(0,frame0.shape[0]-1), random.randint(0,frame0.shape[1]-1), 1/numParticles)


    while(cap.isOpened()):
        ret, frame = cap.read()

        start = time.time()

        ##################################################
        # Calculate Optical Flow
        ##################################################
        nextFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevFrame, nextFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx , dy = flow[...,0], flow[...,1]
        prevFrame = nextFrame

        ##################################################
        # Mask R-CNN Detection
        ##################################################
        results = model.detect([frame], verbose=1)

        r = results[0]
        boxes = r['rois']
        N = boxes.shape[0]

        for i in range(N):
            if r['class_ids'][i] == 1:
                # mean and covariance in the particle filter
                y1, x1, y2, x2 = boxes[i]

                # optical flow center and covariance
                meanX = dx[x1:x2, y1:y2].mean()
                meanY = dy[x1:x2, y1:y2].mean()
                covX = (np.cov(dx[x1:x2, y1:y2])).mean()
                covY = (np.cov(dy[x1:x2, y1:y2])).mean()

                # bounding box center
                boxX = (x1+x2)/2
                boxY = (y1+y2)/2
                covBox = 1 - r['scores'][i]

                # Assign random mean / covariance which needs to be tuned
                # currently using 0.3, 5
                if math.isnan(meanX) :
                    meanX = 0.3
                    covX = 5
                if math.isnan(meanY) :
                    meanY = 0.3
                    covY = 5

                sumX = np.sum(dx[x1:x2, y1:y2])
                sumY = np.sum(dy[x1:x2, y1:y2])

                particles = pf.actionModel(particles, numParticles, boxX, boxY, meanX, meanY, covX, covY)
                particles = pf.sensorModel(particles, numParticles, boxX, boxY)

                for j in range(numParticles):
                    ax.scatter([particles[j].x], [particles[j].y])
                break

        ##################################################
        # Image Plotting
        ##################################################
        visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], "Real Time Detection", size, ax, fig)

        print('Time elapsed: ',time.time() - start)

        if cv2.waitKey(25) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    '''
    Block for IP Streaming

    url = "35.3.71.126:8080"
    link = 'http://' + url + '/video'
    print('Streaming from: ' + link)

    ctr = 0

    stream = urllib2.urlopen(link)
    bytes = bytes()

    # Read until video is completed
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            ctr += 1

            if ctr%10==0:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                start = time.time()
                results = model.detect([frame], verbose=1)
                r = results[0]
                print(time.time() - start)


                ##################################################
                # Image Plotting
                ##################################################
                visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], "Real Time Detection", size, ax, fig)

                # Press esc on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == 27:
                    exit(0)
    '''

