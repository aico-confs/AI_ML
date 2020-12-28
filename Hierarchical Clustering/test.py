# K-Means 與階層式分群法都是基於距離的演算法，這類型的分群演算法適合群間距離單純的資料。
# 如果遇到不能單純使用距離來計算群間的相似度時

from sklearn import datasets                # 載入 scikit-learn 套件中的資料集模組，用來產生資料集
from sklearn import cluster                 # 載入 scikit-learn 套件中的分群演算法模組
import numpy as np
import matplotlib.pyplot as plt

# 產生半月形資料
moon_data, moon_data_labels = datasets.make_moons(n_samples=1500, noise=0.05)
plt.scatter(moon_data[:, 0], moon_data[:, 1], c=moon_data_labels)

# 利用階層式分群法分群行不通
hierarchical = cluster.AgglomerativeClustering(n_clusters=2)
y_pred_h = hierarchical.fit_predict(moon_data)
plt.scatter(moon_data[:, 0], moon_data[:, 1], c=y_pred_h[:])

# 試試其它linkage參數是否能正確分群
hierarchical = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
y_pred_h = hierarchical.fit_predict(moon_data)
plt.scatter(moon_data[:, 0], moon_data[:, 1], c=y_pred_h[:])
plt.show()