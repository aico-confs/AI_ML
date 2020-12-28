from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 讀入鳶尾花資料
iris = load_iris()
# print(iris)
# iris.csv
#2. 切分訓練與測試資料
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=50)
# x_train.shape

#3. 初始化 knn 分類模組
knn = KNeighborsClassifier()  # 預設 K 值為5
iris_knn=knn.fit(x_train,y_train)

#4. 預測
y_test_predicted = iris_knn.predict(x_test)
print(y_test_predicted)                         # 預測的分類
print(y_test)                                   # 標準答案的分類
print(accuracy_score(y_test, y_test_predicted)) # 正確率