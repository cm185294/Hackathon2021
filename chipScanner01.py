from PIL import Image
import PIL
import numpy as np
import cv2
import os
from statistics import mode, mean
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================================================================
# Crop image to chip slice 
def cropImage(imageArray):
    cropAr = []
    newAr = imageArray

    imgHeight = len(newAr)
    imgWidth = len(newAr[0])

    # Crop height
    cropTop = 0.1
    cropBtm = 0.5
    newImgHeight = imgHeight * cropTop
    newImgHeightBtm = imgHeight * cropBtm

    # Pull desired rows from full image
    inc = 0
    for row in newAr:
        if (inc >= newImgHeight and inc <= newImgHeightBtm):
            cropAr.append(row)
        if (inc > newImgHeightBtm):
            break
        inc += 1
        
    return cropAr


# Half the size of the image
def reduceImageSize(imageArray):
    reducedArray = []
    reduceSize = 12

    incCol = 0
    incRow = 0
    for row in imageArray:
        newRow = []
        if (incRow % reduceSize == 0):
            incCol = 0
            for column in row:
                if (incCol % reduceSize == 0):
                    newRow.append(column)
                incCol += 1
            reducedArray.append(newRow)
        incRow += 1

    return reducedArray


# Convert image to grayscale
def convertImageToGray(imageArray):
    newAr = imageArray

    for eachRow in newAr:
        for eachPix in eachRow:
            redG = (eachPix[0] / 255) / 3
            greenG = (eachPix[1] / 255) / 3
            blueG = (eachPix[2] / 255) / 3
            grayscaleVal = (redG + greenG + blueG) * 255

            eachPix[0] = grayscaleVal
            eachPix[1] = grayscaleVal
            eachPix[2] = grayscaleVal

    return newAr


# Threshold the image (chip should stand out)
def threshold(imageArray):
    balanceModeAr = []
    balanceMeanAr = []
    newAr = imageArray

    # Generate average value arrays for mean and mode    
    for eachRow in imageArray:
        for eachPix in eachRow:
            avgNum = mode(eachPix[:3])
            balanceModeAr.append(avgNum)

            avgNum = mean(eachPix[:3])
            balanceMeanAr.append(avgNum)

    # Calculate mean and mode thresholds with offsets
    balanceMode = mode(balanceModeAr) - (mode(balanceModeAr) / 32)
    balanceMean = mean(balanceMeanAr) + (mean(balanceMeanAr) / 4)

    # Set new output colours in grayscale based on threshold
    whiteVal = 127
    for eachRow in newAr:
        for eachPix in eachRow:
            outputVal = 0
            
            if mode(eachPix[:3]) > balanceMode:
                outputVal += whiteVal

            if mean(eachPix[:3]) > balanceMean:
                outputVal += whiteVal

            eachPix[0] = outputVal
            eachPix[1] = outputVal
            eachPix[2] = outputVal
                
    return newAr


# Convert image array and save it
def convertImageArray(imgArray, saveDirectory, imgName):
    print("Crop")
    imgArray = cropImage(imgArray)
    print("Reduce Size")
    imgArray = reduceImageSize(imgArray)
    print("Grayscale")
    imgArray = convertImageToGray(imgArray)
    #print("Threshold")
    #imgArray = threshold(imgArray)

    print("Convert to array")
    imgArray = np.asarray(imgArray)

    print("Save")
    newImage = Image.fromarray(imgArray)
    newImage.save(saveDirectory + '/Generated/' + imgName)
    print()




# ==========================================================================================
# Compare new images with existing dataset of reference cards with chips
def CompareToReferenceSet(originalDir, referenceDir):
    percentageArray = []

    # Get process start time
    startTime = datetime.now()

    imgTotal = 0
    
    for oImage in os.listdir(originalDir):
        if (os.path.isfile(os.path.join(originalDir, oImage))):
            print("Comparing : " + oImage)
            oImg = Image.open(originalDir + '/' + oImage)
            oImg = np.asarray(oImg)
            totalPix = len(oImg) * len(oImg[0])

            rImageCount = 0
            for rImage in os.listdir(referenceDir):
                #print(rImage)
                count = 0
                if (os.path.isfile(os.path.join(referenceDir, rImage))):
                    rImg = Image.open(referenceDir + '/' + rImage)
                    rImg = np.asarray(rImg)

                    rowInc = 0
                    for row in oImg:
                        colInc = 0
                        for col in row:
                            if (oImg[rowInc][colInc][0] == rImg[rowInc][colInc][0]):
                                count += 1
                            colInc += 1
                        rowInc += 1
                rImageCount += 1
                imgTotal += 1
                percent = (count / totalPix * 100)
                #print(percent)
                percentageArray.append(percent)

    accurateCount = 0
    for val in percentageArray:
        if (val > 0.5):
            #print(str(val) + "%")
            accurateCount += 1

    # Get process end time, calculate the total process time and print to log
    endTime = datetime.now()
    processLength = endTime - startTime
    print("Process Took - " + str(processLength) + "ms")

    accuracy = accurateCount / imgTotal * 100

    if (accuracy > 55):
        print("Card has a chip with - " + str(accuracy) + '% confidence')
    else:
        print("No chip on card")


        

CompareToReferenceSet('dataset/videos/ReferenceSet/dataset/TEST',
                      'dataset/videos/ReferenceSet/dataset/Generated')





    

# ==========================================================================================
# Check for Generated directory and create if non-existant
def createGeneratedDirectory(directoryPath):
    # Check if Generated directory already exists
    generatedDirExists = False
    for dirs in os.listdir(directoryPath):
        if (os.path.isdir(os.path.join(directoryPath, dirs))):
            if(dirs == 'Generated'):
                generatedDirExists = True
                break

    # If Generated directory does not exists, create one
    if (generatedDirExists):
        print('Generated Directory Already Exists - No Directory Created')
    else:
        os.mkdir(directoryPath + '/Generated')
        print('Generated Directory Created')
    print()


# Read and extract files, saving converted images
def readAndConvertVideo(directoryPath, filename):
    count = 0
    newFrameCount = 0
    skipFrames = 1

    # Try read file, if it does't exists or is not a video then throw error
    cap = cv2.VideoCapture(os.path.join(directoryPath, filename))
    ret, frame = cap.read()
    if (ret == False):
        print("ERROR - File selected does not exist or is not a video file")
        return False

    fileNameOutput,_ = os.path.splitext(filename)

    # While returning frames, convert and save them
    while(ret):
        ret, frame = cap.read()
        # Skip frames to reduce time
        if (ret and (count % skipFrames == 0)):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("Frame " + str(newFrameCount))
            convertImageArray(np.asarray(frame),
                              directoryPath,
                              fileNameOutput + '-' + str(newFrameCount) + '.png')
            newFrameCount += 1
        count += 1

    cap.release()

    return True


# Read all images in images in a directory, convert it and save new image
def convertDirectoryImages(directoryPath):
    path = directoryPath

    # Get process start time
    startTime = datetime.now()

    # Create Genenerated directory if one doesn't already exist
    createGeneratedDirectory(path)

    successfulImages = 0
    failedImages = []
        
    # Loop through every image in the folder
    for files in os.listdir(path):
        if (os.path.isfile(os.path.join(path, files))):
            print(str(datetime.now()) + " -- Converting: " + files)
            # Try to generate a cropped/grayscaled/thresholded image and save to Generated dir
            try:
                img = Image.open(directoryPath + '/' + files)
                convertImageArray(np.asarray(img), directoryPath, files)

                print("Image converted successfully")
                successfulImages += 1
            except:
                print("Image failed to convert")
                failedImages.append(files)
            print()

    # Get process end time, calculate the total process time and print to log
    endTime = datetime.now()
    processLength = endTime - startTime
    print("Process Took - " + str(processLength) + "ms")
    print("Successful Images Converted: " + str(successfulImages))
    print("Failed to Convert Images   : " + str(len(failedImages)))
    for failed in failedImages:
        print("   - " + failed)


# Convert from video file
def convertDirectoryVideos(directoryPath, filename = ""):       
    # Get process start time
    startTime = datetime.now()

    # Create Genenerated directory if one doesn't already exist
    createGeneratedDirectory(directoryPath)

    # Convert single image if filename given, otherwise convert entire folder
    if (filename == ""):
        successfulVideos = 0
        failedVideos = []
        
        for files in os.listdir(directoryPath):
            if (os.path.isfile(os.path.join(directoryPath, files))):
                print(str(datetime.now()) + " -- Converting: " + files)
                if (readAndConvertVideo(directoryPath, files)):
                    print("Video " + files + " successfully converted")
                    successfulVideos += 1
                else:
                    print("Video " + files + " failed to convert")
                    failedVideos.append(files)
                print()
    else:
        readAndConvertVideo(directoryPath, filename)

    # Get process end time, calculate the total process time and print to log
    endTime = datetime.now()
    processLength = endTime - startTime
    print("Process Took - " + str(processLength) + "ms")

    if (filename == ""):
        print("Successful Videos Converted: " + str(successfulVideos))
        print("Failed to Convert Videos   : " + str(len(failedVideos)))
        for failed in failedVideos:
            print("   - " + failed)









#convertDirectoryVideos('dataset/more_videos/TEST', 'card02.mp4')
#convertDirectoryVideos('dataset/videos/ReferenceSet/dataset')
#convertDirectoryImages('dataset/Cards/DemoVideo')










# ==========================================================================================
def showFour():
    i1 = Image.open('dataset-pre/with01.png')
    iar1 = np.asarray(i1)

    i2 = Image.open('dataset-pre/with01.png')
    iar2 = np.asarray(i2)

    i3 = Image.open('dataset-pre/with02.png')
    iar3 = np.asarray(i3)

    i4 = Image.open('dataset-pre/with02.png')
    iar4 = np.asarray(i4)


    iar1 = threshold(iar1)
    #iar2 = threshold(iar2)
    iar3 = threshold(iar3)
    #iar4 = threshold(iar4)


    fig = plt.figure()
    ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
    ax2 = plt.subplot2grid((8,6), (4,0), rowspan=4, colspan=3)
    ax3 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)
    ax4 = plt.subplot2grid((8,6), (4,3), rowspan=4, colspan=3)

    ax1.imshow(iar1)
    ax2.imshow(iar2)
    ax3.imshow(iar3)
    ax4.imshow(iar4)

    plt.show()

def showOneWithNormal():
    i1 = Image.open('dataset-pre/with02Cropped.png')
    iar1 = np.asarray(i1)
    iar2 = np.asarray(i1)

    iar1 = cropImage(iar1)
    iar2 = cropImage(iar2)

    iar2 = threshold(iar2)

    fig = plt.figure()
    ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
    ax2 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)

    ax1.imshow(iar1)
    ax2.imshow(iar2)
    plt.show()

def showOne():
    i = Image.open('dataset-pre/with02.png')
    iar = np.asarray(i)

    iar = cropImage(iar)
    iar = convertImageToGray(iar)
    iar = threshold(iar)

    plt.imshow(iar)
    plt.show()


#showOne()
#showOneWithNormal()
#showFour()









