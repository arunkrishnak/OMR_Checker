import argparse
import cv2
import numpy as np

def calculate_contour_features(contour):
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)

def calculate_corner_features(corner_file):
    corner_img = cv2.imread(corner_file)
    imgGray=cv2.cvtColor(corner_img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    corner_img_gray = cv2.Canny(imgBlur,10,50)

    contours, hierarchy = cv2.findContours(
        corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 2:
        raise RuntimeError(
            'Did not find the expected contours when looking for the corner')

    corner_contour = next(ct
                          for i, ct in enumerate(contours)
                          if hierarchy[0][i][3] != -1)

    return calculate_contour_features(corner_contour)

def normalize(im):
    imgGray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    return cv2.Canny(imgBlur,10,50)

def get_approx_contour(contour, tol=.01):
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def get_contours(image_gray):
    contours, _ = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return map(get_approx_contour, contours)

def get_top_left_corner(contours):
    top_left_features = calculate_corner_features('Lcorner.jpg')
    
    # Find the contour that matches the top-left corner features
    sorted_contours = sorted(
        contours,
        key=lambda c: features_distance(
            top_left_features,
            calculate_contour_features(c))
    )
    
    # Assuming the best match is the top-left corner
    top_left_corner = sorted_contours[0]
    
    return top_left_corner

def get_other_corners(contours):
    # Calculate the features for the other corners (sqr.jpg)
    other_corner_features = calculate_corner_features('sqr.jpg')
    
    # Find the contours that match the other corners' features
    sorted_contours = sorted(
        contours,
        key=lambda c: features_distance(
            other_corner_features,
            calculate_contour_features(c))
    )
    
    # Assuming the next best matches are the other three corners
    other_corners = sorted_contours[:3]
    
    return other_corners

def get_all_corners(contours):
    # Get the top-left corner
    top_left_corner = get_top_left_corner(contours)
    
    # Remove the top-left corner from the list of contours
    remaining_contours = [c for c in contours if not np.array_equal(c, top_left_corner)]
    
    # Get the other three corners
    other_corners = get_other_corners(remaining_contours)
    
    # Return the top-left corner and the other corners
    return [top_left_corner] + other_corners

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

def sort_points_counter_clockwise(points):
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)

def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    return np.int32(box)

def rotate_to_top_left(outmost, lcorner):
    # Find the index of the Lcorner in the sorted points
    lcorner_idx = np.argmin([np.linalg.norm(point - lcorner) for point in outmost])
    
    # Rotate the points to make Lcorner the first (top-left)
    return np.roll(outmost, -lcorner_idx, axis=0)

def perspective_transform(img, points, width, height):
    source = np.array(points, dtype="float32")
    dest = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]],
        dtype="float32")

    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (width, height))
    return warped

def detect_corners(source_file):
    im_orig =source_file
    
    # Check if the image was loaded successfully
    if im_orig is None:
        raise FileNotFoundError(f"Unable to load the image from {source_file}. Please check the file path.")
    
    height, width = im_orig.shape[:2]
    if width > height:
        height, width = width, height

    im_normalized = normalize(im_orig)
    contours = list(get_contours(im_normalized))
    
    # Get all corners (top-left and other three corners)
    corners = get_all_corners(contours)

    lcorner = get_top_left_corner(contours)
    
    # Draw and transform the image as before
    #cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)
    outmost = sort_points_counter_clockwise(get_outmost_points(corners))
    outmost = rotate_to_top_left(outmost, lcorner[0][0])

    color_transf = perspective_transform(im_orig, outmost, width, height)
    
    return color_transf, im_orig

def croppedOMR(img):
    transformed_img, original_img = detect_corners(img)
    return transformed_img
    # if args.output:
    #     #cv2.imwrite(args.output, original_img)
    #     cv2.imwrite(args.output, transformed_img)
    #     print('Wrote image to {}.'.format(args.output))

    # if args.show:
    #     #cv2.imshow('Original image with detected corners', original_img)
    #     #cv2.imshow('Transformed image', transformed_img)
    #     cv2.waitKey(0)
