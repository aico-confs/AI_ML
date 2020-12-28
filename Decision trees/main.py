from IPython.display import Image
import pydotplus
from sklearn import tree
import pandas as pd
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz\bin\\'
# os.environ["PATH"] += os.pathsep + r'C:\Users\q1233\OneDrive\Desktop\程式自學\AI_編程\AI _ML演算法\linear regression\venv\Lib\site-packages\pydotplus'

# 1. 將csv檔讀入pandas的DataFrame
df = pd.read_csv('Decision_Tree_data.csv')

# 2. factorize函式將Series中的字串，映射為一組數字，相同的字串映射為相同的數字
df['age'], _ = pd.factorize(df['age'])
df['income'], _ = pd.factorize(df['income'])
df['student'], _ = pd.factorize(df['student'])
df['buy_computer'], target_names = pd.factorize(df['buy_computer'])

# 3. 訓練資料及分類答案
x_train = df[['age', 'income', 'student']]
y_train = df[['buy_computer']]

# 4. 建立決策樹分類模組
dtree = tree.DecisionTreeClassifier()
buyPc_dtree = dtree.fit(x_train, y_train)

# 5. 畫出決策樹
feature_names = x_train.columns
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png(), width=800, height=800)
graph.write_pdf('iris_2.pdf')
