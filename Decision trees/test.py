from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus
from IPython.display import Image
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz\bin\\'
#1. 讀入鳶尾花資料
iris = load_iris()
# print(iris.data) 150個樣本，每個樣本4個特徵
# print(iris.target)

#2. 切分訓練與測試資料
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

#3. 建立決策樹分類模組
iris_dtree = tree.DecisionTreeClassifier()
buyPc_dtree = iris_dtree.fit(x_train, y_train)

#4. 預測
y_test_predicted = iris_dtree.predict(x_test)
print(y_test_predicted)                         # 預測的分類
print(y_test)                                   # 標準答案的分類
print(accuracy_score(y_test, y_test_predicted)) # 正確率




#5. 畫出決策樹
dot_data = tree.export_graphviz(iris_dtree, out_file=None,
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png()) # gini屬性用於測量節點的純度，如果一個節點包含的都是同一類，此節點gini=0
graph.write_pdf('iris_2.pdf')

