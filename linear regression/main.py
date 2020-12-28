from sklearn.linear_model import LinearRegression
import numpy as np 
import matplotlib.pyplot as plt

#1. 產生模擬資料
x = np.linspace(0, 5, 50)    # 取0~5間，均勻的取50個點
y = 1.2*x + 0.8 + np.random.randn(50) # randn(50)產生50個常態分布的亂數(平均值0，標準差1)，增加noise
plt.scatter(x, y)            # scatter產生散點圖
plt.plot(x,1.2*x+0.8,'r')    # 畫出原本的完美函數

#2. 用線性迴歸做預測
X=x.reshape(50,1)            # x為[x1,x2,....,x50](1x50的陣列)，但SKlearn需要[[x1],[x2],....,[x50]](50x1的陣列)
regr = LinearRegression()    # 初始化線性迴歸模組
regr.fit(X,y)                # 讓線性迴歸模組學習，模組的變數名稱.fit(輸入資料，正確答案)
y_predicted=regr.predict(X)  # y_predicted為線性迴歸模組預測的結果

plt.scatter(x,y)
plt.plot(x,y_predicted,'b')  # 藍線畫出預測的結果
plt.show()                   # Google Colab不用