from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time


path1 = 'data/household_power_consumption_1000.txt'
# 读取数据
df = pd.read_csv(path1, sep=';', low_memory=False)
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')
# 查看指标相关信息
# print(datas.describe().T)
# print(new_df.head())

def date_format(dt):
    #时间字符串是根据数据来定义的
    # print(''.join(dt))
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return  (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
# iloc根据行和列来获取数据
X = datas.iloc[:, 0:2]
# print(X.iloc[0:1, :])
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
# print(X)
Y = datas['Global_active_power']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train.describe().T)

## 处理数据，将数据转换为标准差为1的数据集
ss = StandardScaler()
# 处理数据，将要训练的数据输入到系统中，系统会自动调整参数来适应训练的数据
X_train = ss.fit_transform(X_train)
# 此时的数据转换，是在上一步参数设置好的情况下，进行的转化，保持数据转换规则一直
X_test = ss.transform(X_test)
# print(pd.DataFrame(X_train).describe().T)
# print(pd.DataFrame(X_test).describe().T)

# 模型训练
lr = LinearRegression()
lr.fit(X_train, Y_train)

# 预测
y_predict = lr.predict(X_test)

print("准确率", lr.score(X_train, Y_train))

# 保存标准化模型
joblib.dump(ss, 'data_ss.model')
# 保存模型
joblib.dump(lr, 'data_lr.model')

#加载标准化模型
# ss = joblib.load('data_ss.model')
#加载模型
# lr = joblib.load('data_lr.model')

t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='real')
plt.plot(t, y_predict, 'g-', linewidth=2, label='predict')
plt.legend(loc='upper left')
plt.title('the relationship between time and power', fontsize=20)
plt.grid(b=True)
plt.show()



