import flask
import werkzeug
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from skimage import exposure, feature, transform, color
import joblib
#from random import randrange
import datetime
#from imutils.object_detection import non_max_suppression
#from detect import object_detection as od
from detection import get_object
import warnings
warnings.filterwarnings('ignore')

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    return prediction(filename)

def prediction(filename):
    ft = 'Combined'
    print(filename)
    image, crop, p1, p2 = get_object(filename)
    if p1 == (0,0) and p2 == (0,0):
        return "Aucun panneau detecté" + "==" + "no sign detected" 
#    im = Image.fromarray(crop)
#    im.save(filename)
#    img = plt.imread(filename)
    # plt.imshow(img)
    # plt.show()
    
    #img = np.array(img)
    else:
        X = get_features([crop], ft)
        X = np.array(X).astype("float")
        #X= X.reshape(-1, 1)
        #print(X.shape)
        # img = img.reshape(784)
        predicted_label = loaded_model.predict(X)
        # print(d[1])
        pred = int(predicted_label[0])
        res = str(pred) + "==" + str(p1[0]) + "==" + str(p1[1]) + "==" + str(p2[0])+ "==" + str(p2[1]) + "==" + str(image.shape[0]) + "==" + str(image.shape[1])
    return res


def extract_HOG(img):
    (H, hog_img) = feature.hog(img, orientations=9, pixels_per_cell=(5,5),
    cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
    hog_img = exposure.rescale_intensity(hog_img, out_range=(0, 255)).astype("uint8")
    return H

def extract_LBP(img):
    rows, cols = img.shape
    radius = 2
    n_points = radius * 8
    lbp_sum=[]
    rows_ = rows+(6-rows%6)
    cols_ = cols+(6-cols%6)
    I1 = np.zeros((rows_,cols_))
    I1[0:rows,0:cols] = img
    for i in range(6):
        for j in range(6):
            img_block = I1[7*i:7*(i+1),7*j:7*(j+1)]
            lbp = feature.local_binary_pattern(img_block, n_points, radius, 'uniform')
            lbp2 = lbp.astype(np.int32)
            max_bins = 59
            train_hist, _ = np.histogram(lbp2.ravel(), normed=True, bins=max_bins, range=(0, max_bins))
            lbp_sum=lbp_sum + train_hist.tolist()
    lbp_sum = np.array(lbp_sum)          
    return lbp_sum


def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

def cart_to_log_polar(img, T):
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    maxRadius = math.hypot(w/2,h/2)
    m = w / math.log(maxRadius)
    polar_img = cv2.logPolar(img,(w/2, h/2), m/T, cv2.WARP_FILL_OUTLIERS+ cv2.INTER_LINEAR)
    return polar_img

def get_features(X, f):
    
    print("[INFO] Extraction of Features...")
    start_time = datetime.datetime.now()
    feat = []
    for i in range(len(X)):
        # show an update every 1,000 images
        if i > 0 and i % 10 == 0:
            print("[INFO] processed {}/{}".format(i, len(X)))
        I = X[i]
        grayim = color.rgb2gray(I)
        grayim = transform.resize(grayim,(40,40))
        grayim = cart_to_log_polar(grayim,0.8)
        if f == 'HoG':
            feat.append(extract_HOG(grayim))
        elif f == 'LBP':
            feat.append(extract_LBP(grayim))
        else:
            feat.append(np.hstack([extract_HOG(grayim), extract_LBP(grayim)]))
        # save the features using numpy save with .npy extention 
        # which reduced the storage space by 4times compared to pickle
    end_time = datetime.datetime.now()
    dt = end_time - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)    
    print("total_time : ",t)
    # feat = np.array(feat)
    # print(feat.shape)
    return feat

if __name__ == "__main__":
    loaded_model = joblib.load("./model.pkl.gz") 
    app.run(host="0.0.0.0", port=5000, debug=True)
