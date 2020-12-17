import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('D:\LN\Python\Projects\digits.png', 0);

#lấy chữ bằng cách cắt từ ảnh dùng để nhận diện, có thể thêm 1 ảnh chứa só khác để nhận diện
chuviet = [np.hsplit(row, 100) for row in np.vsplit(img, 50)] 

x = np.array(chuviet)
trainData = x[:,:50].reshape(-1,400).astype(np.float32);
test = x[:,50:100].reshape(-1,400).astype(np.float32);

k = np.arange(10);
train_labels = np.repeat(k, 250)[:,np.newaxis];

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, train_labels)
kq1, kq2, kq3, kq4 = knn.findNearest(test, 5)
print (kq2)



#print (train)
#cv.imwrite('c1c.jpg', train);
#cv.imwrite('c2c.jpg', trains);
#cv.waitKey(0)
#cv.destroyAllWindows()
