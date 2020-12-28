import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#1. 讀入鳶尾花資料
iris = load_iris()
# print(iris.DESCR)
# iris.data[0]

#2. 切分訓練與測試資料
X = iris.data
Y = iris.target
X = X[:, 2:]  # 為畫圖方便，簡化只取花瓣(Petal)的長度、寬度來預測
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50) # random_state=x 確保每次切分資料的結果都相同
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train) 畫出原本分類圖

#3. 建立 SVC 分類模組
clf = SVC()                     # 初始化 SVC 分類模組，模組的變數名稱=SVC()
# 核函式預設為’rbf’，更改為「‘linear’ 、 ‘poly’ 、 ‘sigmoid’」
clf = SVC(kernel='rbf')
clf.fit(X, Y)                    # 讓分類模組學習，機器的變數名稱.fit(輸入資料，正確答案)

predicted=clf.predict([[2.5, 3]])# 讓分類模組做預測[2.5,3]為那一類
x=np.append(X, [[2.5, 3]], axis=0) # 將[2.5,3]新增至x。當axis沒有值時，會將x重構為一個一維陣列，再將要合併的值新增到它的後面，最終結果是一個一維的陣列
y=np.append(Y,predicted)        # 將分類結果新增至y
plt.scatter(X[:, 0], X[:, 1], c=Y)

y_test_predicted = clf.predict(x_test)

#4. 畫出預測的分類結果
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test_predicted)          # 分類結果，準還是不準？
# plt.scatter(x_test[:,0], x_test[:,1], c=y_test_predicted - y_test) # c=y_test_predicted - y_test 會讓預測同正確分類的為0，不正確為非0
#5. 以網格為輸入，完整畫出分類結果
g_x0, g_x1 = np.meshgrid(np.arange(0, 7, 0.02), np.arange(0,3,0.02))
g_predicted = clf.predict(np.c_[g_x0.ravel(), g_x1.ravel()])             # np.c_ 將 grid_x0、grid_x1 的數值以zip方式組成list，給分類器分類
g_predicted = g_predicted.reshape(g_x0.shape)                            # 將預測的結果(52500, )調整成網點的形狀(150, 350)
plt.contourf(g_x0, g_x1, g_predicted, cmap=plt.cm.coolwarm, alpha=0.7)   # 填充型等⾼線，以預測結果為等高線的高
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()