import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import cv2
import glob

def sliding_window(image, window_size, step_size):
    for row in range(0, image.shape[0], step_size[0]):
        for col in range(0, image.shape[1], step_size[1]):
            yield (row, col, image[row:row + window_size[0], col:col + window_size[1]])

def resize_(img):
    height, width = img.shape[:2]
    if height > width:
        baseheight = 340
        hpercent = baseheight/float(height)
        wsize = int((float(width)*float(hpercent)))
        img = cv2.resize(img, (wsize, baseheight))
    else:
        basewidth = 340
        wpercent = (basewidth/float(width))
        hsize = int((float(height)*float(wpercent)))
        img = cv2.resize(img, (basewidth, hsize))

    return img


def find_object(test_image, clf):
    scale = 0
    detections = []
    downscale=1.25
    window_size = (64, 64)
    step_size = (12, 12)
    for test_image_pyramid in pyramid_gaussian(test_image, downscale=downscale, multichannel=False):
        if test_image_pyramid.shape[0] < window_size[0] or test_image_pyramid.shape[1] < window_size[1]:
            break
        for (row, col, sliding_image) in sliding_window(test_image_pyramid, window_size, step_size):
            if sliding_image.shape != window_size:
                continue
            sliding_image_hog = hog(sliding_image)
            sliding_image_hog = sliding_image_hog.reshape(1, -1)
            pred = clf.predict(sliding_image_hog)
           
            if pred==1:
                pred_prob = clf.decision_function(sliding_image_hog)
                (window_height, window_width) = window_size
                detections.append((int(col*downscale**scale), int(row*downscale**scale),
                        pred_prob[0], int(window_width*downscale**scale), int(window_height*downscale**scale)))
        scale+=1
    
    return detections

def draw_boxes(image, detections):
    if len(detections) == 0:
        return (0,0), (0,0)
    else:
        test_image_detect = image.copy()
        h, w = test_image_detect.shape[:2]
        for detection in detections:
            col = detection[0]
            row = detection[1]
            width = detection[3]
            height = detection[4]
            cv2.rectangle(test_image_detect, pt1=(col, row), pt2=(col+width, row+height), color=(255, 0, 0), thickness=2)
    #    plt.title('before NMS')
    #    plt.imshow(test_image_detect)
    #    plt.show()
        detection = max(detections,key=lambda item:item[2])
        col = detection[0]
        row = detection[1]
        width = detection[3]
        height = detection[4]
        p1 = (max(10,col), max(10,row))
        p2 = (min(w-10,col+width), min(h-10, row+height))

    return p1, p2

def get_object(img):
    
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

    if image.shape[0] > 340:
        image = resize_(image)
    im = image.copy()
    test_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clf = joblib.load('svm_detect.model')
    detections = find_object(test_image, clf)
    p1, p2 = draw_boxes(test_image, detections)
    if p1 == (0,0) and p2 == (0,0):
        return image, image, p1, p2
    else:
        cv2.rectangle(image, p1, p2, color=(0, 255, 0), thickness=4)
    #    plt.title('after NMS')
    #    plt.imshow(image)
    #    plt.axis('off')
    #    plt.show()
        return image, im[p1[1]:p2[1], p1[0]:p2[0]], p1, p2

def main():

    for i in glob.glob('./*.png'):
        print("========================  "+ i.split('\\')[-1] +"  ========================")
        image, crop, _, _ = get_object(i)
    
        plt.imshow(crop)
        plt.axis('off')
        plt.show() 
        
if __name__ == "__main__" :
    main()