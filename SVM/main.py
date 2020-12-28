import pit as pit
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

#1. 分類資料準備
x = np.array([[-3,2], [-6,5], [3,-4], [2,-8]]) # 平面上4個點
x = np.array([[-2,-6], [-2,2], [-4,-4], [2,-4]])
y = np.array([1,1,2,2])                        # y為分屬的類別

plt.scatter([-3, -6, 3, 2], [2, 5, -4, -8], c=[1,1,2,2]) # scatter畫圖時，第一個參數為點的x座標list(array)，第二個參數為點的y座標list(array)。c為指定顏色，不同類別不同色。
plt.scatter(x[:,0], x[:,1], c=y)                         # 上一行的簡化。x[:,0] 表示所有列的第0欄，即所有點的x座標。

#2. 建立 SVC 分類模組
clf = SVC()                     # 初始化 SVC 分類模組，模組的變數名稱=SVC()
# 核函式預設為’rbf’，更改為「‘linear’ 、 ‘poly’ 、 ‘sigmoid’」
clf = SVC(kernel='rbf')
clf.fit(x, y)                    # 讓分類模組學習，機器的變數名稱.fit(輸入資料，正確答案)

predicted=clf.predict([[2.5, 3]])# 讓分類模組做預測[2.5,3]為那一類
x=np.append(x, [[2.5, 3]], axis=0) # 將[2.5,3]新增至x。當axis沒有值時，會將x重構為一個一維陣列，再將要合併的值新增到它的後面，最終結果是一個一維的陣列
y=np.append(y,predicted)        # 將分類結果新增至y
plt.scatter(x[:, 0], x[:, 1], c=y)

#3. 完整畫出分類結果
X, Y= np.meshgrid(np.linspace(-6, 3, 30), np.linspace(-8, 5, 30))    # 以meshgrid產生格點
X=X.ravel()            # 將產生的2維陣列轉成1維陣列
Y=Y.ravel()
plt.scatter(X, Y)       # 畫出平面上的所有點(先試)

predicted=clf.predict(list(zip(X, Y)))
plt.scatter(X, Y, c=predicted)
plt.show()