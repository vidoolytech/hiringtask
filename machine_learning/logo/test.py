import cv2
import numpy as np
from sklearn.externals import joblib


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

map_dict = {
    "Adidas" : 1,
    "Apple" : 2,
    "BMW" : 3,
    "Citroen" : 4,
    "Cocacola" : 5,
    "DHL" : 6,
    "Fedex" : 7,
    "Ferrari" : 8,
    "Ford" : 9,
    "Google" : 10,

}

clf = joblib.load("model400.pkl")
km = joblib.load("cluster400.pkl")

test = "logos/flickr_logos_27_dataset_query_set_annotation.txt"
imgpath = "logos/flickr_logos_27_dataset_images/"
accuracy = []

f = open(test)

for line in f.readlines():
    imgfile, logoClass = line.split("\t")
    #imgfile, logoClass, subset, x1, y1, x2, y2, _  = line.split(" ")
    logoClass = logoClass.split("\r")[0]

    img = cv2.imread(imgpath + imgfile)
    if logoClass == "Heineken":
        break

    print imgfile, logoClass
    #edge = cv2.Canny(img, 150, 200)

    fd = cv2.xfeatures2d.SIFT_create()

    keypoint, des = fd.detectAndCompute(img, None)
    if des is not None:
        p = km.predict(des)

        result = clf.predict([np.bincount(p, minlength=400)])
        print result[0]
        accuracy.append(result[0] == (map_dict[logoClass]))
        #cv2.imshow("image", edge)
        #cv2.waitKey(0)
print "Accuracy: " + str(mean(accuracy)*100)
