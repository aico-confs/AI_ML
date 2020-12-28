from sklearn import datasets                # 載入 scikit-learn 套件中的資料集模組，用來產生資料集
from sklearn import cluster                 # 載入 scikit-learn 套件中的分群演算法模組
import numpy as np
import matplotlib.pyplot as plt

#1. 分群資料準備，隨機產生4群資料
points, label = datasets.make_blobs(n_samples=1500, centers=4, cluster_std=3, random_state=170) # make_blobs()隨機產生一群一群圓形的資料。cluster_std為群中資料點離群中央的距離標準差，值愈大資料愈分散。random_state為隨機種子。
plt.scatter(points[:, 0], points[:, 1], c=label[:])          # 畫出產生的點，依據不同群給予不同顏色(先試)

#2. 以 Agglomerative Clustering 分群模組做分群
hierarchical = cluster.AgglomerativeClustering(n_clusters=4) # 初始化 AgglomerativeClustering 分群模組
y_pred_h = hierarchical.fit_predict(points)                  # 開始分群

#3.  畫出分群結果
plt.scatter(points[:, 0], points[:, 1], c=y_pred_h[:])
plt.show()