import numpy as np
import cv2
import math 

from numpy.random import seed
from numpy.random import randint

from tqdm import tqdm
import numpy as np


def generateImageAndLabel(imageSize, objectsArrayShape):
    
    width, height = imageSize
    imageShape = (height, width, 1)

    # przerobić macierz kodującą na float z zakresu [0.0 - 1.0]
    arr = np.reshape(randint(0, 4, objectsArrayShape[0]* objectsArrayShape[1] ), objectsArrayShape)
    tileHeight = height // objectsArrayShape[0]
    tileWidth = width // objectsArrayShape[1]

    img = np.zeros(imageShape, np.uint8)

    for j, row in enumerate(arr):
        for i, element in enumerate(row):
            
            element = int(element)
            
            #square
            if (element is 0):
                offsetWidth = tileWidth//8
                offsetHeight = tileHeight//8
                topLeft = (i*tileHeight + offsetHeight, j*tileWidth + offsetWidth)
                bottomRight = (i*tileHeight + tileHeight - offsetHeight , j*tileWidth + tileWidth - offsetWidth)
                cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), cv2.FILLED)
            
            #circle
            elif element is 1:
                center = (i*tileHeight + tileHeight//2 ,j*tileWidth + tileWidth//2)
                radiusOffset = tileWidth//8
                cv2.circle(img, center, tileWidth//2 - radiusOffset, (255, 255, 255), cv2.FILLED)
            #triangle
            elif element is 2:
                center = (i*tileHeight + tileHeight//2 ,j*tileWidth + tileWidth//2)
                Point1 = [center[0], center[1] - tileHeight//2 + tileHeight//8]
                Point2 = [center[0] - tileWidth//2 + tileWidth//8, center[1] + tileHeight//2 - tileHeight//8]
                Point3 = [center[0] + tileWidth//2 - tileWidth//8, center[1] + tileHeight//2 - tileHeight//8]
                pts = np.array([Point1, Point2, Point3 ], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(img, [pts], True, (255, 255 ,255))
                cv2.fillPoly(img, [pts], (255, 255, 255))
            
            elif element is 4:
                # dodać kreskę poziomą
                pass
            elif element is 5:
                # dodać kreskę pionową
                pass
            elif element is 6:
                pass
            
            #default empty
            else:
                pass


    return img, arr.reshape(-1)

def generateImage_SingleShape(imageSize, shapeID):
    width, height = imageSize
    imageShape = (height, width, 1)
    img = np.zeros(imageShape, np.uint8)

    margin = 0.1 # fraction of width/height

    minWidth = width // 4
    minHeight = height // 4

    lineThickness = max(1, width//32)


    # square (lines only)
    if shapeID is 0:

        squareMaxDimension = min(width - math.ceil(width*(margin*2)), height - math.ceil(height*(margin*2)))
        squareDimmension = randint(min(minWidth, minHeight), squareMaxDimension)

        topLeft = (randint(math.ceil(width*margin), width - math.ceil(width*margin) - squareDimmension), 
                    randint(math.ceil(height*margin), height - math.ceil(height*margin) - squareDimmension) )
        
        bottomRight = tuple(sum(x) for x in zip(topLeft, (squareDimmension, squareDimmension)))
        cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), thickness=lineThickness)


    #square (filled)
    elif shapeID is 1:

        squareMaxDimension = min(width - math.ceil(width*(margin*2)), height - math.ceil(height*(margin*2)))
        squareDimmension = randint(min(minWidth, minHeight), squareMaxDimension)

        topLeft = (randint(math.ceil(width*margin), width - math.ceil(width*margin) - squareDimmension), 
                    randint(math.ceil(height*margin), height - math.ceil(height*margin) - squareDimmension) )
        
        bottomRight = tuple(sum(x) for x in zip(topLeft, (squareDimmension, squareDimmension)))
        cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), cv2.FILLED)

    # circle
    elif shapeID is 2:

        maxRadius = min(width - math.ceil(width*(margin*2)), height - math.ceil(height*(margin*2))) //2
        radius = randint((min(minWidth, minHeight)), maxRadius)

        center = (randint(math.ceil(width*margin) + radius, width - math.ceil(width*margin) - radius), 
            randint( radius + math.ceil(height*margin), height - math.ceil(height*margin) - radius) )

        cv2.circle(img, center, radius, (255, 255, 255), thickness=lineThickness)
    
    # circle
    elif shapeID is 3:

        maxRadius = min(width - math.ceil(width*(margin*2)), height - math.ceil(height*(margin*2))) //2
        radius = randint((min(minWidth, minHeight)), maxRadius)

        center = (randint(math.ceil(width*margin) + radius, width - math.ceil(width*margin) - radius), 
            randint( radius + math.ceil(height*margin), height - math.ceil(height*margin) - radius) )

        cv2.circle(img, center, radius, (255, 255, 255), cv2.FILLED)
    
    #triangle
    elif shapeID is 4:
        
        maxSize = min(width - math.ceil(width*(margin*2)), height - math.ceil(height*(margin*2)))
        size = randint(10, maxSize)

        center = (randint(math.ceil(width*margin) + size//2, width - math.ceil(width*margin) - size//2), 
            randint( math.ceil(height*margin +  size//2), height - math.ceil(height*margin) - size//2) )
        
        Point1 = [center[0], center[1] - size//2 + size//8]
        Point2 = [center[0] - size//2 + size//8, center[1] + size//2 - size//8]
        Point3 = [center[0] + size//2 - size//8, center[1] + size//2 - size//8]
        pts = np.array([Point1, Point2, Point3 ], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (255, 255 ,255))
        # cv2.fillPoly(img, [pts], (255, 255, 255))
    
    #triangle (filled)
    elif shapeID is 5:
        
        maxSize = min(width - math.ceil(width*(margin*2)), height - math.ceil(height*(margin*2)))
        size = randint(10, maxSize)

        center = (randint(math.ceil(width*margin) + size//2, width - math.ceil(width*margin) - size//2), 
            randint( math.ceil(height*margin +  size//2), height - math.ceil(height*margin) - size//2) )
        
        Point1 = [center[0], center[1] - size//2 + size//8]
        Point2 = [center[0] - size//2 + size//8, center[1] + size//2 - size//8]
        Point3 = [center[0] + size//2 - size//8, center[1] + size//2 - size//8]
        pts = np.array([Point1, Point2, Point3 ], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (255, 255 ,255))
        cv2.fillPoly(img, [pts], (255, 255, 255))
    
    #rectangle (narrow)
    elif shapeID is 6:
        rectangleMaxWidth = (width - math.ceil(width*(margin*2)) )//2
        rectangleMaxHeight = height - math.ceil(height*(margin*2))
        
        
        rectangleWidth = randint(minWidth//2, rectangleMaxWidth)
        rectangleHeight = randint(minHeight, rectangleMaxHeight)


        topLeft = (randint(math.ceil((width*margin)), width - math.ceil(width*margin) - rectangleWidth ), 
                    randint(math.ceil((height*margin)), height - math.ceil(height*margin) - rectangleHeight ))
        
        bottomRight = tuple(sum(x) for x in zip(topLeft, (rectangleWidth, rectangleHeight)))
        
        cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), thickness=lineThickness)

    #rectangle (narrow, filled)
    elif shapeID is 7:
        rectangleMaxWidth = (width - math.ceil(width*(margin*2)) )//2
        rectangleMaxHeight = height - math.ceil(height*(margin*2))
        
        
        rectangleWidth = randint(minWidth//2, rectangleMaxWidth)
        rectangleHeight = randint(minHeight, rectangleMaxHeight)


        topLeft = (randint(math.ceil((width*margin)), width - math.ceil(width*margin) - rectangleWidth ), 
                    randint(math.ceil((height*margin)), height - math.ceil(height*margin) - rectangleHeight ))
        
        bottomRight = tuple(sum(x) for x in zip(topLeft, (rectangleWidth, rectangleHeight)))
        
        cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), cv2.FILLED)

    # rectangle (short)
    elif shapeID is 8:
        rectangleMaxWidth = width - math.ceil(width*(margin*2))
        rectangleMaxHeight = (height - math.ceil(height*(margin*2))) // 3
        
        
        rectangleWidth = randint(minWidth, rectangleMaxWidth)
        rectangleHeight = randint(minHeight//2, rectangleMaxHeight)


        topLeft = (randint(math.ceil((width*margin)), width - math.ceil(width*margin) - rectangleWidth ), 
                    randint(math.ceil((height*margin)), height - math.ceil(height*margin) - rectangleHeight ))
        
        bottomRight = tuple(sum(x) for x in zip(topLeft, (rectangleWidth, rectangleHeight)))
        
        cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), thickness=lineThickness)
 
    # rectangle (short, filled)
    elif shapeID is 9:
        rectangleMaxWidth = width - math.ceil(width*(margin*2))
        rectangleMaxHeight = (height - math.ceil(height*(margin*2))) // 3
        
        
        rectangleWidth = randint(minWidth, rectangleMaxWidth)
        rectangleHeight = randint(minHeight//2, rectangleMaxHeight)


        topLeft = (randint(math.ceil((width*margin)), width - math.ceil(width*margin) - rectangleWidth ), 
                    randint(math.ceil((height*margin)), height - math.ceil(height*margin) - rectangleHeight ))
        
        bottomRight = tuple(sum(x) for x in zip(topLeft, (rectangleWidth, rectangleHeight)))
        
        cv2.rectangle(img, topLeft, bottomRight, (255, 255 , 255), cv2.FILLED)

    else:
        a = 6
        pass

    return img

def generateLabeledDataset(quantity, imageSize, objectsArrayShape):
    imagesList = []
    labelsList = []
    for i in tqdm(range(quantity)):
        image, label = generateImageAndLabel(imageSize, objectsArrayShape)
        imagesList.append(image)
        labelsList.append(label)
    
    images = np.array(imagesList)
    labels = np.array(labelsList)
    return images, labels

def generateLabeledDataset_SingleShape(quantity, imageSize):
    imagesList = []
    labelsList = randint(0, 11, size=(quantity, 1 ))
    for i, id in tqdm(enumerate(labelsList)):
        image = generateImage_SingleShape(imageSize, int(id) )
        imagesList.append(image)
        # cv2.imshow("test", image)
        # cv2.waitKey(300)
    
    images = np.array(imagesList)

    return images, labelsList


if __name__ == "__main__":
    
    
    # images, labels = generateLabeledDataset(10000, (64, 64), (3, 3))
    images, labels = generateLabeledDataset_SingleShape(10000, (28, 28))

    
    # imagesList = []
    # for i in range(1000):
    #     image = generateImage((64, 64), (3, 3))
    #     imagesList.append(image)
    #     cv2.imshow("test", image)
    #     cv2.waitKey(10)
   
    # imagesList = []
    # for i in tqdm(range(10000)):
    #     image = generateImage_SingleShape((28, 28), 5)
    #     cv2.imshow("test", image)
    #     cv2.waitKey(50)



    pass