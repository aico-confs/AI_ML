from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs             # 載入 scikit-learn 套件中的資料集模組，用來產生資料集
import numpy as np
import matplotlib.pyplot as plt

#1. 分群資料準備，隨機產生4群資料
points, label = make_blobs(n_samples=1500, centers=4, cluster_std=3, random_state=170) # make_blobs()隨機產生一群一群圓形的資料。cluster_std為群中資料點離群中央的距離標準差，值愈大資料愈分散。random_state為隨機種子。
plt.scatter(points[:, 0], points[:, 1], c=label[:]) # 畫出產生的點，依據不同群給予不同顏色(先試)

#2. 用K-Means做分群
clf = KMeans(n_clusters=4)              # 初始化 KMeans 分群模組，n_clusters為要分成幾群，試試看其它數字
y_predicted = clf.fit_predict(points)   # 用 K-Means 分群

#3. 畫出分群結果
plt.scatter(points[:, 0], points[:, 1], c=y_predicted[:])
plt.show()