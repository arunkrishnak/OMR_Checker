import csv
import cv2
import numpy as np
import utils
import fitz  # PyMuPDF
import cornerdetect

###############################################
path="output.pdf"
widthImg=2480
heightImg=3508
Sectionq=20
qchoices=6
lnameq=26
lchoices=13
fnameq=26
fchoices=7
mnameq=26
mchoices=1
sidq=10
sidchoices=14
svq=10
svchoices=14
###############################################    

img=cv2.imread(path)
 
# Convert PDF to image using PyMuPDF
def pdf_to_image(pdf_path, page_number=0, zoom=1):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)  # Load the specified page
    
    # Define zoom factor
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)  # Render page to image with zoom factor
    pdf_document.close()
    
    return pix.tobytes()

# Convert the PDF to an image (first page)
zoom_factor = 3
img_bytes = pdf_to_image(path, zoom=zoom_factor)
img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
img=cornerdetect.croppedOMR(img)
#cv2.imwrite("path.jpg", img)

#Image Processing
img=cv2.resize(img,(widthImg,heightImg))


######################## SECTION 1 QUESTIONS ###############################

# APPLY THRESHOLD
trim_top = 2134
trim_left = 160
trim_right = 1966
trim_bottom = 54
# Apply Trimming
s1trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("s1trimed.jpg", s1trimed)
#cv2.imshow("trimed",s1trimed)

imgWarpGray = cv2.cvtColor(s1trimed,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)


boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[58])
# cv2.imshow("Last",boxes[119])

myPixelVal=np.zeros((Sectionq,qchoices))#arrayfilled with zeros
countC=0
countR=0
for image in boxes:
    totalPixels=cv2.countNonZero(image)
    myPixelVal[countR][countC]=totalPixels
    countC+=1
    if(countC==qchoices):
        countR+=1
        countC=0
#print(myPixelVal)

sec1records=[]
for row in range(Sectionq):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 750:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        sec1records.append(str(myIndexVal[0][0]+ 1))
    else:
        sec1records.append("Not marked")
print("sec1records")
print(sec1records)


# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## SECTION 2 QUESTIONS ###############################
# APPLY THRESHOLD
trim_top = 2134
trim_left = 624
trim_right = 1502
trim_bottom = 54
# Apply Trimming
s2trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("s2trimed.jpg", s2trimed)
#cv2.imshow("trimed",s2trimed)

imgWarpGray = cv2.cvtColor(s2trimed,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

#height, width = imgThresh.shape
# print("Height of Script Version:", height)
#print("Width of Script Version:", width)


boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[56])
# cv2.imshow("Last",boxes[119])

myPixelVal=np.zeros((Sectionq,qchoices))#arrayfilled with zeros
countC=0
countR=0
for image in boxes:
    totalPixels=cv2.countNonZero(image)
    myPixelVal[countR][countC]=totalPixels
    countC+=1
    if(countC==qchoices):
        countR+=1
        countC=0
#print(myPixelVal)

sec2records=[]
for row in range(Sectionq):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        sec2records.append(str(myIndexVal[0][0]+ 1))
    else:
        sec2records.append("Not marked")
print("sec2records")
print(sec2records)

# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec2records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## SECTION 3 QUESTIONS ###############################

# APPLY THRESHOLD
trim_top = 2134
trim_left = 1110
trim_right = 1016
trim_bottom = 54
# Apply Trimming
s3trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("s3trimed.jpg", s3trimed)
#cv2.imshow("trimed",s3trimed)

imgWarpGray = cv2.cvtColor(s3trimed,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[57])
# cv2.imshow("Last",boxes[119])

myPixelVal=np.zeros((Sectionq,qchoices))#arrayfilled with zeros
countC=0
countR=0
for image in boxes:
    totalPixels=cv2.countNonZero(image)
    myPixelVal[countR][countC]=totalPixels
    countC+=1
    if(countC==qchoices):
        countR+=1
        countC=0
#print(myPixelVal)

# FIND THE USER ANSWERS AND PUT THEM IN A LIST
sec3records=[]
for row in range(Sectionq):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        sec3records.append(str(myIndexVal[0][0]+ 1))
    else:
        sec3records.append("Not marked")
print("sec3records")
print(sec3records)

# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")

#########################################################################

######################## SECTION 4 QUESTIONS ###############################
# APPLY THRESHOLD
trim_top = 2134
trim_left = 1563
trim_right = 563
trim_bottom = 54
# Apply Trimming
s4trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("s4trimed.jpg", s4trimed)
#cv2.imshow("trimed",s4trimed)

imgWarpGray = cv2.cvtColor(s4trimed,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[57])
# cv2.imshow("Last",boxes[119])

myPixelVal=np.zeros((Sectionq,qchoices))#arrayfilled with zeros
countC=0
countR=0
for image in boxes:
    totalPixels=cv2.countNonZero(image)
    myPixelVal[countR][countC]=totalPixels
    countC+=1
    if(countC==qchoices):
        countR+=1
        countC=0
#print(myPixelVal)

# FIND THE USER ANSWERS AND PUT THEM IN A LIST
sec4records=[]
for row in range(Sectionq):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        sec4records.append(str(myIndexVal[0][0]+ 1))
    else:
        sec4records.append("Not marked")
print("sec4records")
print(sec4records)

# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")

#########################################################################

######################## SECTION 5 QUESTIONS ###############################
# APPLY THRESHOLD
trim_top = 2134
trim_left = 2022
trim_right = 104
trim_bottom = 54

# Apply Trimming
s5trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("s5trimed.jpg", s5trimed)
#cv2.imshow("trimed",s5trimed)

imgWarpGray = cv2.cvtColor(s5trimed,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[57])
# cv2.imshow("Last",boxes[119])

myPixelVal=np.zeros((Sectionq,qchoices))#arrayfilled with zeros
countC=0
countR=0
for image in boxes:
    totalPixels=cv2.countNonZero(image)
    myPixelVal[countR][countC]=totalPixels
    countC+=1
    if(countC==qchoices):
        countR+=1
        countC=0
#print(myPixelVal)

# FIND THE USER ANSWERS AND PUT THEM IN A LIST
sec5records=[]
for row in range(Sectionq):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        sec5records.append(str(myIndexVal[0][0]+ 1))
    else:
        sec5records.append("Not marked")
print("sec5records")
print(sec5records)

# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## LAST NAME SECTION ###############################

# APPLY THRESHOLD
trim_top = 377
trim_left = 71
trim_right = 1642
trim_bottom = 1857

# Apply Trimming
lname = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("lname.jpg", lname)
#cv2.imshow("trimed",lname)

imgWarpGray = cv2.cvtColor(lname,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,lnameq,lchoices)

transposed_boxes=[[boxes[row * lchoices + col] for row in range(lnameq)] for col in range(lchoices)]
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[169])
# cv2.imshow("Last",boxes[337])

myPixelVal=np.zeros((lchoices,lnameq))#arrayfilled with zeros

for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels
                
# FIND THE USER ANSWERS AND PUT THEM IN A LIST
lnamerecords=[]
for row in range(lchoices):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        lnamerecords.append(str(myIndexVal[0][0]+ 1))
    else:
        lnamerecords.append("Not marked")
print("Lname")
print(lnamerecords)


# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## FIRST NAME SECTION ###############################

# APPLY THRESHOLD
trim_top = 377
trim_left = 937
trim_right = 1137
trim_bottom = 1857

# Apply Trimming
fname = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("fname.jpg", fname)
#cv2.imshow("trimed",fname)

imgWarpGray = cv2.cvtColor(fname,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,fnameq,fchoices)

transposed_boxes=[[boxes[row * fchoices + col] for row in range(fnameq)] for col in range(fchoices)]
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[90])
# cv2.imshow("Last",boxes[181])

myPixelVal=np.zeros((fchoices,fnameq))#arrayfilled with zeros

for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels
                
# FIND THE USER ANSWERS AND PUT THEM IN A LIST
fnamerecords=[]
for row in range(fchoices):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        fnamerecords.append(str(myIndexVal[0][0]+ 1))
    else:
        fnamerecords.append("Not marked")
print("fname")
print(fnamerecords)
"""
# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## MID NAME SECTION ###############################
"""
# APPLY THRESHOLD
trim_top = 377
trim_left = 1408
trim_right = 1024
trim_bottom = 1857

# Apply Trimming
mname = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("mname.jpg", mname)
#cv2.imshow("trimed",fname)

imgWarpGray = cv2.cvtColor(mname,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitVerticallyBoxes(imgThresh,mnameq)
# cv2.imshow("test1",boxes[0])
# cv2.imshow("test2",boxes[1])
# cv2.imshow("test3",boxes[2])
# cv2.imshow("test4",boxes[3])
# cv2.imshow("Mid",boxes[12])
# cv2.imshow("Last",boxes[25])

myPixelVal=np.zeros((mnameq,mchoices))#arrayfilled with zeros
countC=0
countR=0
for image in boxes:
    totalPixels=cv2.countNonZero(image)
    myPixelVal[countR][countC]=totalPixels
    countC+=1
    if(countC==1):
        countR+=1
        countC=0
#print(myPixelVal)

# FIND THE USER ANSWERS AND PUT THEM IN A LIST
mnamerecords=[]
for row in range(mnameq):
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        mnamerecords.append(str(myIndexVal[0][0]+ 1))
    else:
        mnamerecords.append("Not marked")
print("Mname")
print(mnamerecords)

# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## STUDENT ID SECTION ###############################

# APPLY THRESHOLD
trim_top = 378
trim_left = 1602
trim_right = 80
trim_bottom = 2580
# Apply Trimming
stid= img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("stid.jpg", stid)
#cv2.imshow("trimed",stid)

imgWarpGray = cv2.cvtColor(stid,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,sidq,sidchoices)

transposed_boxes=[[boxes[row * sidchoices + col] for row in range(sidq)] for col in range(sidchoices)]
# cv2.imshow("test1",boxes[6])
# cv2.imshow("test2",boxes[13])
# cv2.imshow("test3",boxes[19])
# cv2.imshow("test4",boxes[27])
# cv2.imshow("Mid",boxes[34])
# cv2.imshow("Last",boxes[139])

myPixelVal=np.zeros((sidchoices,sidq))#arrayfilled with zeros

for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels

# FIND THE USER ANSWERS AND PUT THEM IN A LIST
sidrecords=[]
for row in range(sidchoices):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        sidrecords.append(str(myIndexVal[0][0]))
    else:
        sidrecords.append("Not marked")
print("Student ID")
print(sidrecords)

# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

######################## SCRIPT VERSION SECTION ###############################

# APPLY THRESHOLD
trim_top = 1104
trim_left = 1605
trim_right = 77
trim_bottom = 1854
# Apply Trimming
sversion= img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
cv2.imwrite("sversion.jpg", sversion)
#cv2.imshow("trimed",sversion)

imgWarpGray = cv2.cvtColor(sversion,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
imgThresh = cv2.threshold(imgWarpGray, 180, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
#cv2.imshow("Thresh",imgThresh)

# height, width = imgThresh.shape
# print("Height of Script Version:", height)
# print("Width of Script Version:", width)

boxes=utils.splitBoxes(imgThresh,svq,svchoices)

transposed_boxes=[[boxes[row * svchoices + col] for row in range(svq)] for col in range(svchoices)]
# cv2.imshow("test1",boxes[6])
# cv2.imshow("test2",boxes[13])
# cv2.imshow("test3",boxes[19])
# cv2.imshow("test4",boxes[27])
# cv2.imshow("Mid",boxes[34])
# cv2.imshow("Last",boxes[139])

myPixelVal=np.zeros((svchoices,svq))#arrayfilled with zeros

for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels

# FIND THE USER ANSWERS AND PUT THEM IN A LIST
scriptVrecords=[]
for row in range(svchoices):#after transposing the array have 8 rows
    max_val = np.amax(myPixelVal[row])
    #print(max_val)
    if max_val > 700:
        myIndexVal = np.where(myPixelVal[row] == max_val)
        scriptVrecords.append(str(myIndexVal[0][0]))
    else:
        scriptVrecords.append("Not marked")
print("Script Version")
print(scriptVrecords)
"""
# Save records to CSV
# with open(filename, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Questions", "Answers"])
#     for i, answer in enumerate(sec1records, start=1):
#         writer.writerow([f"Q{i}", answer])
#     writer.writerow("")
#########################################################################

imgRawDrawings = np.zeros_like(imgCornerPionts)
imageArray=([img,imgGray,imgBlur,imgCanny],
        [imgRawDrawings,imgCornerPionts,imgRawDrawings,imgRawDrawings])#second row of stack 
imageStacked=utils.stackImages(imageArray,0.5)

cv2.imshow("stacked",imageStacked)
"""
cv2.waitKey(0)
