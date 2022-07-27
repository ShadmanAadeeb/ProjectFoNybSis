from msilib.schema import ComboBox
from tkinter import *
from tkinter.ttk import Combobox
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy


# this function will be called when I will try to apply my model on an image
def useModel(net, colors, img):
    #************ keeping a copy of the original image along with its width and height
    originalImage=copy.deepcopy(img)
    height, width, _ = img.shape

    # Create image of the same size of the input image and placing black color everywhere
    # black_image = np.zeros((height, width, 3), np.uint8)
    # black_image[:] = (0, 0, 0)


    #********* Creating a blob (preprocessed image) and getting the boxes and masks
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    # placing the blob into the model
    net.setInput(blob)

    # forwarding the blob and getting boxes and masks
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    boxes_count = boxes.shape[2]
    # boxes will give (1,1,100,7) . i.e 100 boxes with 7 values
    # print(f'boxes shape = {boxes.shape}')
    # print(f'boxes values {boxes[0][0][0]}')
    # print(f'mask shape = {masks.shape}')  # there are a total of 100 masks


    # FOR THE UI I WILL BE USING THE FOLLOWING
    # selectedBoxes for storing the regions where the boxes have been found
    # selectedMasks for storing the regions where the segments have been found
    # objectIds for storing the objects
    # a dictionary mentioning how many class ids are there for this object, it will help in creating the objectids / names
    
    #**********These will be used for storing the results
    labelsPath = 'labels.txt'
    LABELS = open(labelsPath).read().strip().split("\n")
    selectedBoxes = []
    selectedMasks = []
    objectIds = []
    classCountDict = dict()
    mappingDict = {i: value for i, value in enumerate(LABELS)}
    # print(mappingDict)


    #***********iterating over the boxes
    for i in range(boxes_count):
    # get a box from the 100 boxes
        
        #***********getting a box
        box = boxes[0, 0, i]
    # print(f'SHAPE OF MY BOX = {box.shape}')

    # get the class id of the box and the score of the box, if it is below thresh, then ifnore
        
        #************collecting its class id
        class_id = box[1]

        #*************score based rejection
        score = box[2]
        if score < 0.5:
            continue

    # *************Get box Coordinates from the 3rd, 4th, 5th and 6th position of the box and scale it up accordngly
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

    # getting that region in the black image and also storing that shape
        # roi = black_image[y: y2, x: x2]
        # roi_height, roi_width, _ = roi.shape

    # Get the masks, each of the hundred boxes have a mask associated with it
    # Masks are available for 90 classes , 15 by 15 values are there for each class

    # amra ei box er oi mask take nicchi
    
    #*******************getting the mask related to this class     
        mask = masks[i, int(class_id)]
    # print(mask)
    # etake resize korchi black image er shoman kore
        mask = cv2.resize(mask, (x2-x, y2-y))
    # then we are thresholding it
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

    # placing the box on the image
        # cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

    # detecting the contours in the mask
        # contours, _ = cv2.findContours(
        # np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # color = colors[int(class_id)]


    
    # placing the contours in the roi, which will automatically place in the black image
    
        # for cnt in contours:        
        #     cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))    
        
    
    # i have the following now    
    # a mask marking where the region is
    # x,y,x2,y2 marking the boxed region
    # class_id
        # print(i)
        
        #******************storing the findings (boxes,masks,object ids)
        selectedBoxes.append((x,y,x2,y2))
        selectedMasks.append(mask)
        if(class_id not in classCountDict):
            classCountDict[class_id]=1
        else:
            classCountDict[class_id]=classCountDict[class_id]+1    
        objectIds.append(''+mappingDict[class_id]+str(classCountDict[class_id]))
    
    # cv2.imshow("roi", roi)
        # cv2.waitKey(0)
    #cv2.imshow('originalImage',originalImage)

    return originalImage,selectedBoxes,selectedMasks,objectIds,classCountDict


# the global vars
currentImage=None


root = Tk()
root.title('Picture Segmentation Project For Nybsys')


# preparing the functions for the Select button
def selectAndProcess():
    
    global currentImage
    global container
    # print('select an image')
    # open the filedialog to collect filepath
    filepath = filedialog.askopenfilename(initialdir="./", title="Select An Image", filetypes=(("jpeg files", "*.jpg"), ("gif files", "*.gif*"), ("png files", "*.png")))
    
    #*************opening a file if it is selected
    if(len(filepath)!=0):
        # keeping this image
        currentImage=cv2.imread(filepath)
        # keeping a blurred copy of this image
        #gausBlur = cv2.GaussianBlur(currentImage, (51,51),0) 

        #*************** now applying the model by calling my prepared function and getting the results
        net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        # Generate random colors (80 color arrays with 3 channels)
        colors = np.random.randint(0, 255, (80, 3))

        
        global originalImage
        global selectedBoxes
        global selectedMasks
        global objectIds
        global classCountDict

        originalImage, selectedBoxes, selectedMasks, objectIds, classCountDict = useModel(net, colors, currentImage)
        # print(f'orginal image = { originalImage.shape}')
        # print(f'selected  boxes = { selectedBoxes}')
        # print(f'selected  masks = { len(selectedMasks)}')
        # print(f'obejctIds = { objectIds}')
        # print(f'obejctIds = { classCountDict}')                
        
        #******* displaying this (SELECTED) image (parameter->currentImageForContainer) ke image ta dite hobe
        currentImageForContainer=copy.deepcopy(originalImage)
        currentImageForContainer = cv2.cvtColor(currentImageForContainer, cv2.COLOR_BGR2RGB)        
        pilImage = Image.fromarray(np.uint8(currentImageForContainer)).convert('RGB')
        #pilImage = Image.open('blackScreen.jpg')
        photoImage = ImageTk.PhotoImage(pilImage.resize((600, 350)))        
        container.configure(image=photoImage)
        container.image=photoImage

        #***********updating the combobox
        global comboBox
        newList=['None']
        for objectId in objectIds:
            newList.append(objectId)
        comboBox['values'] = newList        


#************** preparing the SELECT BUTTON widget
selectButton = Button(root, text='Select', width=30, bg='yellow', fg='red',command=selectAndProcess)

#************** preparing the IMAGE CONTAINING LABEL widget
cv2Image=cv2.imread('blackScreen.jpg')
pilImage = Image.fromarray(np.uint8(cv2Image)).convert('RGB')
#pilImage = Image.open('blackScreen.jpg')
photoImage = ImageTk.PhotoImage(pilImage.resize((600, 350)))
container = Label(root, image=photoImage)

#**************** preparing the COMBOBOX widget
comboBox=Combobox(root)
comboBox["values"]=("None")
comboBox['state'] = 'readonly'

# creating and binding a function with the combobox widget
def comboBoxSelectionFunction(event):
    global originalImage
    global selectedBoxes
    global selectedMasks
    global objectIds
    global classCountDict
    
    value=event.widget.get()
    
    # jodi user None select kore then we just set the blurred image
    if(value=="None"):
        # print(f'You selected the value {value}')
        
        gausBlur = cv2.GaussianBlur(originalImage, (51,51),0)
        # displaying this image (parameter->currentImageForContainer) ke image ta dite hobe
        currentImageForContainer=copy.deepcopy(gausBlur)
        currentImageForContainer = cv2.cvtColor(currentImageForContainer, cv2.COLOR_BGR2RGB)        
        pilImage = Image.fromarray(np.uint8(currentImageForContainer)).convert('RGB')
        #pilImage = Image.open('blackScreen.jpg')
        photoImage = ImageTk.PhotoImage(pilImage.resize((600, 350)))        
        container.configure(image=photoImage)
        container.image=photoImage
        return 

    # print("******************************************")
    # print(f'orginal image = { originalImage.shape}')
    # print(f'selected  boxes = { selectedBoxes}')
    # print(f'selected  masks = { len(selectedMasks)}')
    # print(f'obejctIds = { objectIds}')
    # print(f'obejctIds = { classCountDict}')      
    
    # oi object er index ber kori
    selectedObjectIndex=objectIds.index(value)
    # print("******************************************")
    # print(f'The index of the selected object is {selectedObjectIndex}')

    # preparing the processed output
    #extracting the mask bits and no mask bits region
    currentBox=selectedBoxes[selectedObjectIndex]
    currentMask=selectedMasks[selectedObjectIndex]    
    x1,y1,x2,y2=currentBox
    currentBoxExtracted=copy.deepcopy(originalImage)[y1:y2,x1:x2]
    maskBits=np.where(currentMask==1)
    nonMaskBits=np.where(currentMask==0)

    gausBlur = cv2.GaussianBlur(originalImage, (51,51),0) 

    currentBoxWithObjectExtracted=copy.deepcopy(currentBoxExtracted)
    currentBoxWithObjectExtracted[nonMaskBits]=copy.deepcopy(gausBlur)[y1:y2,x1:x2][nonMaskBits]

    # a blurred image is prepared where selected image is being highlighted 
    
    gausBlur[y1:y2,x1:x2]=currentBoxWithObjectExtracted

    # displaying this image (parameter->currentImageForContainer) ke image ta dite hobe
    currentImageForContainer=copy.deepcopy(gausBlur)
    currentImageForContainer = cv2.cvtColor(currentImageForContainer, cv2.COLOR_BGR2RGB)        
    pilImage = Image.fromarray(np.uint8(currentImageForContainer)).convert('RGB')
    #pilImage = Image.open('blackScreen.jpg')
    photoImage = ImageTk.PhotoImage(pilImage.resize((600, 350)))        
    container.configure(image=photoImage)
    container.image=photoImage


comboBox.bind("<<ComboboxSelected>>", comboBoxSelectionFunction)


# placing the container and the button and the combo box
container.grid(column=0, row=0, columnspan=4)
selectButton.grid(column=1, row=1,columnspan=2)
comboBox.grid(column=1, row=2,columnspan=2)




root.mainloop()

