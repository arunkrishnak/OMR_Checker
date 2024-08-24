import numpy as np
import cv2
def splitBoxes(img, rows, cols):
    # Calculate the size of each box
    h, w = img.shape[:2]
    box_h = h // rows
    box_w = w // cols

    # Ensure box dimensions are integers
    box_h = int(box_h)
    box_w = int(box_w)

    # Create list to hold the boxes
    boxes = []

    for r in range(rows):
        for c in range(cols):
            top = r * box_h
            bottom = (r + 1) * box_h
            left = c * box_w
            right = (c + 1) * box_w
            
            # Handle cases where the last box might exceed image dimensions
            if r == rows - 1:
                bottom = h
            if c == cols - 1:
                right = w

            box = img[top:bottom, left:right]
            boxes.append(box)

    return boxes
def splitVerticallyBoxes(img,vsplits):
    rows = np.vsplit(img,vsplits)
    boxes=[]
    for r in rows:
        boxes.append(r)   
    return boxes

# Define constants
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def align_and_crop_image(input_image, ref_image_path="cropped.jpg"):
    """
    Aligns the input image to the reference image and crops out black borders.
    
    Parameters:
    - input_image: numpy array, the input image.
    - ref_image_path: str, path to the reference image (default is "cropped.jpg").
    
    Returns:
    - Cropped, aligned image.
    """
    # Load the reference image
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    # Use the input image directly
    im1 = input_image
    im2 = ref_image

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Check if descriptors were found
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors could not be computed. Possibly no features detected.")

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
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
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    # Crop the image to remove black borders
    gray_mask = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        im1Reg_cropped = im1Reg[y:y+h, x:x+w]
    else:
        im1Reg_cropped = im1Reg

    return im1Reg_cropped
