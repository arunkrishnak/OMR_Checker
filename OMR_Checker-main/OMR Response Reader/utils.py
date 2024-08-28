import numpy as np
import cv2

def splitBoxes(img,vsplits,hsplits):
    rows = np.vsplit(img,vsplits)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,hsplits)
        for box in cols:
            boxes.append(box)
    return boxes

def splitVerticallyBoxes(img,vsplits):
    rows = np.vsplit(img,vsplits)
    boxes=[]
    for r in rows:
        boxes.append(r)   
    return boxes

#   Aligns the input image to the reference image
def align_and_crop_image(input_image, ref_image_path="referenceOMR.jpg"):

    # Define constants
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Load the reference image
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    # Use the input image directly
    im1 = input_image
    im2Gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Check if descriptors were found
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors could not be computed. Possibly no features detected.")

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove bad matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    if numGoodMatches < 4:
        raise ValueError("Not enough good matches were found.")

    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    if h is None:
        raise ValueError("Homography could not be computed.")

    # Warp perspective
    height, width, channels = ref_image.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    _, thresh = cv2.threshold(im1, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        im1Reg_cropped = im1Reg[y:y+h, x:x+w]
    else:
        im1Reg_cropped = im1Reg

    return im1Reg_cropped
