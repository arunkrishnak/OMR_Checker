import numpy as np
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
