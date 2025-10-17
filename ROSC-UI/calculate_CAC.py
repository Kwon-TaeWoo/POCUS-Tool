"""
Code Name  : semifinal_calculate_CAC.py
Author     : Subin Park (subinn.park@gmail.com)
Created on : 24. 2. 7. 오후 1:36
Desc       : 
"""

import cv2
import numpy as np
import os
import pandas as pd
import time # time 라이브러리 import


def measure_motion_blur(image):
    # Convert image to grayscale
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    # Calculate Laplacian variance
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return var

def measure_vog(image):
    # Convert image to grayscale
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    
    # Calculate the standard deviation of the gradient in the x and y directions
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    stdx = np.std(dx)
    stdy = np.std(dy)
    
    # Calculate the VoG as the ratio of the standard deviations
    vog = stdy / stdx
    
    return vog

def channel_check(img):
    # Convert image to grayscale if it's not already
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray


def contour_detection(img, pixel_value):
    # Convert image to grayscale if it's not already
    gray = channel_check(img)
    
    # Binarize the image for the given pixel value
    ret, binary = cv2.threshold(gray, pixel_value-1, pixel_value, cv2.THRESH_BINARY)
    return binary

def remove_small_regions(img, size, pixel_value):
    # Convert image to grayscale if it's not already
    gray = channel_check(img)
    
    # Binarize the image for the given pixel value
    ret, binary = cv2.threshold(gray, pixel_value-1, pixel_value, cv2.THRESH_BINARY)
    # Find all connected components
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a copy of the binary image to manipulate
    output = binary.copy()
    # Loop through each contour

    for contour in contours:
        # If the contour is smaller than size, remove it
        if cv2.contourArea(contour) < size:
            cv2.drawContours(output, [contour], -1, 0, -1)
    # Return the updated image

    return output


def moving_avarage_smoothing(X,k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S


def detect_ellipse(major_axis, minor_axis):
    if major_axis == minor_axis:
        return "This is a circle."
    elif major_axis > minor_axis:
        eccentricity = (1 - (minor_axis / major_axis) ** 2) ** 0.5
        if eccentricity < 0.1:
            return "This is a nearly circular ellipse."
        else:
            return "This is an elliptical shape."
    elif minor_axis > major_axis:
        return "This is an inverted elliptical shape."
    else:
        return "This shape is not an ellipse."

    

def process_images_realtime(file, in_img, img, size, draw_ellipse):
    measurement = [file, 0, 0, 0]
    '''
    #noise check code
    # Compute the Laplacian variance of the image
    laplacian_var = np.var(cv2.Laplacian(in_img, cv2.CV_64F))
    measurement[7] = laplacian_var
        
    # Measure motion blur
    blur = measure_motion_blur(in_img)
    measurement[8] = blur

    '''
    # Measure VoG
    #vog = measure_vog(in_img)
    #measurement[3] = vog

    if len(img.shape) != 2:
        binary_120 = cv2.inRange(img, np.array([120,120,120]), np.array([120,120,120]))
        binary_240 = cv2.inRange(img, np.array([240,240,240]), np.array([240,240,240]))
    else:
        binary_120 = cv2.inRange(img, np.array([120]), np.array([120]))
        binary_240 = cv2.inRange(img, np.array([240]), np.array([240]))
        
    #binary_120 = contour_detection(binary_120, 255)
    #binary_240 = contour_detection(binary_240, 255)
    
    #small area removed
    #binary_120 = remove_small_regions(binary_120,300, 255)
    #binary_240 = remove_small_regions(binary_240,300, 255)
    
    #binary_120[binary_120 == 255] = 120
    #binary_240[binary_240 == 255] = 240
        
    #cv2.imwrite(os.path.join(output_folder, file), binary_120 + binary_240)
        
    contours_120, _ = cv2.findContours(binary_120.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_240, _ = cv2.findContours(binary_240.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Draw the translucent areas on the original image and save it to the output folder
    overlay = in_img.copy()
    alpha = 0.3  # Transparency factor


    if contours_240:
        contour_240 = max(contours_240, key=cv2.contourArea)
        if len(contour_240) >= 5:
            ellipse = cv2.fitEllipse(contour_240)
            #measurement[3] = min(ellipse[1])
            #measurement[4] = max(ellipse[1])
            measurement[2] = (1 - (min(ellipse[1]) / max(ellipse[1])) ** 2) ** 0.5

            #cv2.ellipse(overlay, ellipse, (255, 0, 0), 2)
            #cv2.ellipse(overlay, ellipse, (255, 0, 0), -1, 8)        
    
    if measurement[2] == int(0):
        measurement[2] = int(1)
            
    if contours_120:
        contour_120 = max(contours_120, key=cv2.contourArea)
        #measurement______ = cv2.contourArea(contour_120)
        if len(contour_120) >=5:
            ellipse = cv2.fitEllipse(contour_120)
            #measurement[1] = min(ellipse[1])
            #measurement[2] = max(ellipse[1])
            measurement[3] = (max(ellipse[1]) / min(ellipse[1]))
            measurement[1] = (1 - (min(ellipse[1]) / max(ellipse[1])) ** 2) ** 0.5
                
            if (float(measurement[2]) < 0.95) and (float(measurement[1]) < 0.70):
                if draw_ellipse:
                    cv2.ellipse(overlay, ellipse, (255, 255, 255), 2)
                else:
                    x, y, w, h = cv2.boundingRect(contour_120)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            elif (float(measurement[1]) >= 0.94):
                if draw_ellipse:
                    cv2.ellipse(overlay, ellipse, (0, 0, 255), 2)
                else:
                    x, y, w, h = cv2.boundingRect(contour_120)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
            elif (float(measurement[1]) < 0.94)and (float(measurement[2]) >= 0.95):
                if draw_ellipse:
                    cv2.ellipse(overlay, ellipse, (0, 255, 0), 2)
                else:
                    x, y, w, h = cv2.boundingRect(contour_120)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            #cv2.ellipse(overlay, ellipse, (0, 0, 255), -1, 8)
            

       
    # Apply the translucent overlay using addWeighted function
    #cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


    #measurements.append(measurement)
    return measurement, overlay