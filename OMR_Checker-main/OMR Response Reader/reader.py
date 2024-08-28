import csv
import cv2
import numpy as np
import utils
import fitz  
import os
import glob

def readeromr(path,zoom_factor=3):
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
    alphabet_map = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    alphabet_to_value = {
        'a': '01', 'b': '02', 'c': '04', 'd': '08', 'e': '16', 'f': '32'
    }    

    img=cv2.imread(path)

    # Convert the PDF to an image
    def pdf_to_image(pdf_path, page_number=0, zoom=1):
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pdf_document.close()
        return pix.tobytes()

    img_bytes = pdf_to_image(path, zoom=zoom_factor)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = utils.align_and_crop_image(img)
    img=cv2.resize(img,(widthImg,heightImg))

    ######################## STUDENT ID SECTION ###############################
    # Apply Trimming
    trim_top = 378
    trim_left = 1602
    trim_right = 80
    trim_bottom = 2580
    stid= img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(stid, 180, 255,cv2.THRESH_BINARY_INV )[1]# APPLY THRESHOLD AND INVERSE

    #Splitting the bubbels
    boxes=utils.splitBoxes(imgThresh,sidq,sidchoices)
    #transposed boxes for vertical scanning
    transposed_boxes=[[boxes[row * sidchoices + col] for row in range(sidq)] for col in range(sidchoices)]

    myPixelVal = np.zeros((sidchoices, sidq)) #array filled with 0

    for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels    #Storing pixel values of each bubbels

    sidrecords = []
    for row in range(sidchoices):
        max_val = np.amax(myPixelVal[row])
        if max_val > 700:   #Pixel density above 600 is considered marked
            myIndexVal = np.where(myPixelVal[row] == max_val)
            sidrecords.append(str(myIndexVal[0][0]))
        else:
            sidrecords.append("0")  

    student_id = ''.join(sidrecords)
    #########################################################################

    ######################## SCRIPT VERSION SECTION ###############################
    trim_top = 1104
    trim_left = 1605
    trim_right = 77
    trim_bottom = 1854
    sversion= img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(sversion, 180, 255,cv2.THRESH_BINARY_INV )[1]

    boxes=utils.splitBoxes(imgThresh,svq,svchoices)
    transposed_boxes=[[boxes[row * svchoices + col] for row in range(svq)] for col in range(svchoices)]

    myPixelVal = np.zeros((svchoices, svq))
    for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels

    scriptVrecords = []
    for row in range(svchoices):
        max_val = np.amax(myPixelVal[row])
        if max_val > 700:
            myIndexVal = np.where(myPixelVal[row] == max_val)
            scriptVrecords.append(str(myIndexVal[0][0]))
        else:
            scriptVrecords.append(" ")
    scriptversion_number = ''.join(scriptVrecords)
    #########################################################################

    ######################## LAST NAME SECTION ###############################
    trim_top = 377
    trim_left = 71
    trim_right = 1642
    trim_bottom = 1857
    lname = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(lname, 180, 255,cv2.THRESH_BINARY_INV )[1] 

    boxes=utils.splitBoxes(imgThresh,lnameq,lchoices)
    transposed_boxes=[[boxes[row * lchoices + col] for row in range(lnameq)] for col in range(lchoices)]

    myPixelVal = np.zeros((lchoices, lnameq)) 
    for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels

    lnamerecords = []
    for row in range(lchoices):
        max_val = np.amax(myPixelVal[row])
        if max_val > 700:
            myIndexVal = np.where(myPixelVal[row] == max_val)
            lnamerecords.append(alphabet_map[myIndexVal[0][0]])
        else:
            lnamerecords.append(" ")
    lnamerecords = ''.join(lnamerecords)
    #########################################################################

    ######################## FIRST NAME SECTION ###############################
    trim_top = 377
    trim_left = 937
    trim_right = 1137
    trim_bottom = 1857
    fname = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(fname, 180, 255,cv2.THRESH_BINARY_INV )[1]

    boxes=utils.splitBoxes(imgThresh,fnameq,fchoices)
    transposed_boxes=[[boxes[row * fchoices + col] for row in range(fnameq)] for col in range(fchoices)]

    myPixelVal = np.zeros((fchoices, fnameq))
    for countC, col_images in enumerate(transposed_boxes):
        for countR, image in enumerate(col_images):
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countC][countR] = totalPixels

    fnamerecords = []
    for row in range(fchoices):
        max_val = np.amax(myPixelVal[row])
        if max_val > 700:
            myIndexVal = np.where(myPixelVal[row] == max_val)
            fnamerecords.append(alphabet_map[myIndexVal[0][0]])
        else:
            fnamerecords.append(" ")  

    fnamerecords = ''.join(fnamerecords)
    #########################################################################

    ######################## MID NAME SECTION ###############################
    trim_top = 377
    trim_left = 1408
    trim_right = 1024
    trim_bottom = 1857
    mname = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(mname, 180, 255,cv2.THRESH_BINARY_INV )[1] 

    boxes=utils.splitVerticallyBoxes(imgThresh,mnameq)

    myPixelVal = np.zeros((mnameq, mchoices))  # array filled with zeros
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if(countC == mchoices):
            countR += 1
            countC = 0

    mnamerecords = []
    for row in range(mnameq):
        max_val = np.amax(myPixelVal[row])
        if max_val > 700:
            myIndexVal = np.where(myPixelVal[row] == max_val)
            mnamerecords.append(chr(97 + row))
    if not mnamerecords:
        mnamerecords.append(' ')

    mnameRemoved = ''.join(mnamerecords)
    #########################################################################   

    ######################## SECTION 1 QUESTIONS ###############################
    trim_top = 2134
    trim_left = 160
    trim_right = 1966
    trim_bottom = 54
    s1trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(s1trimed, 180, 255,cv2.THRESH_BINARY_INV )[1] 

    boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)

    myPixelVal = np.zeros((Sectionq, qchoices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == qchoices:
            countR += 1
            countC = 0

    sec1records = []
    for row in range(Sectionq):
        selected_values = []
        for col in range(qchoices):
            if myPixelVal[row][col] > 750:
                letter = alphabet_map[col]          # Get the alphabet corresponding to the column index
                if letter in alphabet_to_value:     # Add the corresponding value to the list
                    selected_values.append(alphabet_to_value[letter])
        
        # Concatenate all selected values or set "00" if none
        if selected_values:
            answer_string = ''.join(selected_values)
        else:
            answer_string = '00'
        
        sec1records.append(answer_string)

    answers = ''.join(sec1records)
    #########################################################################

    ######################## SECTION 2 QUESTIONS ###############################
    trim_top = 2134
    trim_left = 624
    trim_right = 1502
    trim_bottom = 54
    s2trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(s2trimed, 180, 255,cv2.THRESH_BINARY_INV )[1] 

    boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)

    myPixelVal = np.zeros((Sectionq, qchoices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == qchoices:
            countR += 1
            countC = 0

    sec1records = []
    for row in range(Sectionq):
        selected_values = []
        for col in range(qchoices):
            if myPixelVal[row][col] > 750:
                letter = alphabet_map[col]
                if letter in alphabet_to_value:
                    selected_values.append(alphabet_to_value[letter])
        
        if selected_values:
            answer_string = ''.join(selected_values)
        else:
            answer_string = '00'
        
        sec1records.append(answer_string)

    answers2 = ''.join(sec1records)
    #########################################################################

    ######################## SECTION 3 QUESTIONS ###############################
    trim_top = 2134
    trim_left = 1110
    trim_right = 1016
    trim_bottom = 54
    s3trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(s3trimed, 180, 255,cv2.THRESH_BINARY_INV )[1]

    boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)

    myPixelVal = np.zeros((Sectionq, qchoices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == qchoices:
            countR += 1
            countC = 0

    sec1records = []
    for row in range(Sectionq):
        selected_values = []
        for col in range(qchoices):
            if myPixelVal[row][col] > 750:
                letter = alphabet_map[col]
                if letter in alphabet_to_value:
                    selected_values.append(alphabet_to_value[letter])
        
        if selected_values:
            answer_string = ''.join(selected_values)
        else:
            answer_string = '00'
        
        sec1records.append(answer_string)

    answers3 = ''.join(sec1records)

    #########################################################################

    ######################## SECTION 4 QUESTIONS ###############################
    trim_top = 2134
    trim_left = 1563
    trim_right = 563
    trim_bottom = 54
    s4trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(s4trimed, 180, 255,cv2.THRESH_BINARY_INV )[1] 

    boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)

    myPixelVal = np.zeros((Sectionq, qchoices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == qchoices:
            countR += 1
            countC = 0

    sec1records = []
    for row in range(Sectionq):
        selected_values = []
        for col in range(qchoices):
            if myPixelVal[row][col] > 750:
                letter = alphabet_map[col]
                if letter in alphabet_to_value:
                    selected_values.append(alphabet_to_value[letter])
        
        if selected_values:
            answer_string = ''.join(selected_values)
        else:
            answer_string = '00'
        sec1records.append(answer_string)

    answers4 = ''.join(sec1records)
    #########################################################################

    ######################## SECTION 5 QUESTIONS ###############################
    trim_top = 2134
    trim_left = 2022
    trim_right = 104
    trim_bottom = 54
    s5trimed = img[trim_top:img.shape[0] - trim_bottom, trim_left:img.shape[1] - trim_right]
    imgThresh = cv2.threshold(s5trimed, 180, 255,cv2.THRESH_BINARY_INV )[1]

    boxes=utils.splitBoxes(imgThresh,Sectionq,qchoices)

    myPixelVal = np.zeros((Sectionq, qchoices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == qchoices:
            countR += 1
            countC = 0

    sec1records = []
    for row in range(Sectionq):
        selected_values = []
        for col in range(qchoices):
            if myPixelVal[row][col] > 750:
                letter = alphabet_map[col]
                if letter in alphabet_to_value:
                    selected_values.append(alphabet_to_value[letter])
        
        if selected_values:
            answer_string = ''.join(selected_values)
        else:
            answer_string = '00'
        sec1records.append(answer_string)

    answers5 = ''.join(sec1records)
    #########################################################################
    # Define the filename for responses
    filename = 'Response.txt'

    # Csv saved as txt
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{student_id} {scriptversion_number} {lnamerecords} {fnamerecords} {mnameRemoved} {answers}{answers2}{answers3}{answers4}{answers5}"])
    cv2.waitKey(0)
def split_pdf(input_pdf_path, output_folder):
    # Open the input PDF
    pdf_document = fitz.open(input_pdf_path)
    # Get the total number of pages
    total_pages = pdf_document.page_count
    # Process each page
    for page_number in range(total_pages):
        # Create a new PDF for each page
        new_pdf = fitz.open()
        page = pdf_document.load_page(page_number)
        new_pdf.insert_pdf(pdf_document, from_page=page_number, to_page=page_number)
        # Save the single-page PDF
        single_page_pdf_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_pdf_path))[0]}_page_{page_number + 1}.pdf")
        new_pdf.save(single_page_pdf_path)
        new_pdf.close()
    pdf_document.close()

def process_pdfs_in_folder(folder_path, zoom_factor=3):
    pdf_pattern = os.path.join(folder_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    temp_folder = os.path.join(folder_path, "temp_single_page_pdfs")
    os.makedirs(temp_folder, exist_ok=True)
    for pdf_file in pdf_files:
        split_pdf(input_pdf_path=pdf_file, output_folder=temp_folder)
        single_page_pdfs = glob.glob(os.path.join(temp_folder, "*.pdf"))
        for single_page_pdf in single_page_pdfs:
            readeromr(path=single_page_pdf, zoom_factor=zoom_factor)
        for single_page_pdf in single_page_pdfs:
            os.remove(single_page_pdf)
    
    os.rmdir(temp_folder)

process_pdfs_in_folder("answerpaper", zoom_factor=3)
