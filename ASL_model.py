import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

import warnings
import numpy as np
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)


# to remove the background so that only the hand is show
background = None
accumulated_weight = 0.5

# For the dimensions for the ROI (region of interest)
ROI_top, ROI_bottom, ROI_right, ROI_left = 100, 300, 150, 350

'''
For differentiating between the background and the foreground, we will calculate the
accumulated weighted average for the background and then subtract this from the frames
that contain some object in front of the background that can be distinguished as the
foreground

This is done by calculating the accumulated weight for some frames (60 frames) and then
calculating the weighted average for the background

After we have the accumulated avg for the background, we subtract it from every frame
that we read after 60 frames to find any object that covers the background.
'''


def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

# calculate the threshold value
'''
calculate the threshold value for every frame and determine the contors
using cv2.findContours and return the max contours (the outermost contours
for the object) using the function segment.
Using the contours, we are able to determine if there is any foreground object
being detected in the ROI, i.e.there is a hand in the region of interest
'''

def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)


    _, thresholded =  cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Get the external contours for the Image
    # image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

'''
When contours are detected (or hand is present in the ROI), We start to save
the image of the ROI in the train and test set respectively for the letter or
number we are detecting it for.
'''

cam = cv2.VideoCapture(0)
num_frames = 0
element = 10 # chnge this to 1, 2, 3, ... 9, 10
num_imgs_taken = 0

# Save 700 images for each number (should take around 30 seconds) for the training set
# Save around 50 images for each number for the testing set


while True:
    # get footage from the camera
    ret, frame = cam.read()
    # flipping the frame to prevent inverted image of captured frame...

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # add a blur to use less data + make it quicker to train in the model
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)

        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # have the text "fetching background ... please wait" for the first 60 frames
            # so that the accum_weighted_average for the background pixels can be calculated
            # then we can use this to differentiate between a hand and the background

    #Time to configure the hand specifically into the ROI...
    elif num_frames <= 300:
        hand = segment_hand(gray_frame)

        # after the camera has calibrated for the background, then we can take 240 images for
        # the hand for the testing and training datasets

        cv2.putText(frame_copy, f"Adjust hand ... Gesture for {str(element)}", (200, 400),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Checking if the hand is actually detected by counting the number of contours detected
        if hand is not None:

            thresholded, hand_segment = hand
            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
            ROI_top)], -1, (255, 0, 0),1)

            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # Also display the thresholded image
            cv2.imshow("Thresholded Hand Image", thresholded)

    else:

        # Segmenting the hand region...
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:
            # unpack the thresholded img and the max_contour...
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
            ROI_top)], -1, (255, 0, 0),1)

            cv2.putText(frame_copy, str(num_frames), (70, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.putText(frame_copy, f'{str(num_imgs_taken)} images for {str(element)}',
            (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Displaying the thresholded image
            cv2.imshow("Thresholded Hand Image", thresholded)
            if num_imgs_taken <= 50:
                # Save the image into the appropriate folder
                cv2.imwrite(f"test\\{str(element)}\\{str(num_imgs_taken+300)}.jpg", thresholded)

            else:
                break
            num_imgs_taken +=1
        else:
            cv2.putText(frame_copy, 'No hand detected...', (200, 400),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # Drawing ROI on frame copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,128,0), 3)

    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)

    # increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Sign Detection", frame_copy)

    # Closing windows with Esc key...(any other key with ord can be used too.)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Releasing the camera & destroying all the windows...
cv2.destroyAllWindows()
cam.release()
