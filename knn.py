# read input from pickle
# turn them into numpy array
import pickle
import numpy as np
name='msd_data1.pickle'
infile=open(name,'rb')
data=pickle.load(infile)
infile.close()
Y_train=np.array(data['Y_train'])
X_train=np.array(data['X_train'])
Y_test=np.array(data['Y_test'])
X_test=np.array(data['X_test'])
# standarlization
mean=np.mean(X_train, axis=0)
std=np.std(X_train, axis=0)
X_train_norm=X_train.copy()
X_test_norm=X_test.copy()
X_train_norm=(X_train_norm-mean)/std
X_test_norm=(X_test_norm-mean)/std
# 第一題 [myknn_regressor]
# Q1.1 
# Create your myknn_regressor.

# 將index和distance存在一個list of list
# 再用sort取第二個值去比較
def take(x):
    return x[1]
class myknn_regressor:
    def __init__(self, k, option):
        self.k=k
        self.option=option
        self.xtrain=[]
        self.ytrain=[]
    def fit(self, x, y):
        self.xtrain=x
        self.ytrain=y
    def predict(self, xtest):
        ypred=np.zeros((len(xtest),))
        for i in range(len(xtest)):#each test case
            dist=[]#index+distance
            for j in range(len(self.xtrain)):
                temp=(xtest[i]-self.xtrain[j])**(2)
                dist.append([j,np.sum(temp)])
            dist.sort(key=take)
            top=[]
            for j in range(self.k):
                ypred[i]+=self.ytrain[dist[j][0]]/self.k
                top.append(self.ytrain[dist[j][0]])
            if self.option=='remove_outliers' and self.k>=10:
                top.sort()
                q1=self.k*0.25
                if q1 % 1 ==0:
                    q1=int(q1)
                    q1=(top[q1]+top[q1-1])/2
                else:
                    q1=top[int(q1)]
                q3=self.k*0.75
                if q3 % 1 ==0:
                    q3=int(q3)
                    q3=(top[q3]+top[q3-1])/2
                else:
                    q3=top[int(q3)]
                iqr=q3-q1
                low=q1-1.5*iqr
                high=q3+1.5*iqr
                ya=[]
                for x in top:
                    if x>low and x<high:
                        ya.append(x.copy())
                ya=np.array(ya)
                ypred[i]=np.mean(ya)
        return ypred
def rmse(ypred, y):
    return np.sqrt(np.mean((ypred-y)**2))
# Q1.2 
# Predictions using k=20 and "equal_weight" f
# List the RMSE and the first 20 predictions in the testing data.
myknn = myknn_regressor(20, "equal_weight")
myknn.fit(X_train_norm, Y_train)
ypred = myknn.predict(X_test_norm)
performance=rmse(ypred,Y_test)
print(ypred[:20])
print(performance)
# Q1.3
# Predictions using k=20 and "remove_outier" f
# List the RMSE and the first 20 predictions in the testing data.
myknn = myknn_regressor(20, "remove_outliers")
myknn.fit(X_train_norm, Y_train)
ypred = myknn.predict(X_test_norm)
performance=rmse(ypred,Y_test)
print(ypred[:20])
print(performance)
# 第二題 [Tuning the Hyper-parameter]
# case1: knn with normalization
from sklearn.neighbors import KNeighborsRegressor
ks=[1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,80,100,120,140,160,180,200]
rmses_norm=[]
for k in ks:
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train_norm, Y_train)
    knnn=neigh.predict(X_test_norm)
    performance=rmse(knnn,Y_test)
    rmses_norm.append(performance)
    print(performance)
# case2: knn without normalization
rmses=[]
for k in ks:
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, Y_train)
    knn=neigh.predict(X_test)
    performance=rmse(knn,Y_test)
    rmses.append(performance)
    print(performance)
# case3: myknn
my_rmses=[]
for k in ks:
    myknn = myknn_regressor(k, "remove_outliers")
    myknn.fit(X_train_norm, Y_train)
    ypred = myknn.predict(X_test_norm)
    print(ypred)
    performance=rmse(ypred,Y_test)
    my_rmses.append(performance)
    print(performance)
# plot the curves
# relations between  k(x-axis) and RMSE (y-axis)
# red: knn without normalization
# blue: knn with normalization
# greeen: my knn
import matplotlib.pyplot as plt
plt.plot(ks, rmses_norm, 'b')
plt.plot(ks, rmses, 'r')
plt.plot(ks, my_rmses, 'g')
plt.show()
"""
結果討論：

有normalize過的knn表現都比較好，因為normalize後將feature的影響力變得一樣，
譬如要預測性別，有個feature是身高，有個feature是體重，這兩個單位和數值不一樣，
直接拿來計算，會讓一公分跟一公斤的影響力一樣，而實際上不一定如此，
所以普遍來說normalize過後的表比較好。

可能也會有normalize後表現變差的情形，我推論可能是數值較大的剛好是影響力較大的feature，
剛好放大了這個feature的影響力，符合這個特定資料的特性，所以放原本資料的表現還比較好。

自己實作的knn表現比套件好，可能是因為remove oulier的效果，讓預測不受離群值影響，所以準確率提高。
"""