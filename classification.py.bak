# 导入库
import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVR表示回归器，SVC表示分类器
#from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
# 数据准备

#======================================================================
# 这里是读取数据
#raw_data = np.loadtxt('regression.txt')  # 读取数据文件
#X = raw_data[:, :-1]  # 分割自变量
#y = raw_data[:, -1]  # 分割因变量


print("开始读入数据")

# 读入文件,第一行为列名
features = pd.read_csv('5份数据/SRT_TOP.csv',low_memory=False)
# 展示前五行
display(features.head(5))
print("读入数据完成===============================")
# 这里表示取得标签这一列

# Use numpy to convert to arrays
import numpy as np

# label表示ph值，即是 y标签
# np.array() 表示创建一个数组
labels = features['SRT']
# 展示标签值
print(labels)

# 转化成 one-hot编码，把非数字的值转化为数字，即是one-hot编码
#labels = pd.get_dummies(labels)
print("获取标签数据完成==============================================")

# pandas的drop()函数，是用来删除某一行或者某一列。如果axix=1表示删除某一列，默认的axis为0 ，即选择删除某一行。
# 这里表示删除标签这一列

features= features.drop('SRT', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)

print("删除标签列完成==============================================")


#转换成数组
features_array = np.array(features)
print(features_array)
print(features_array.shape)

from sklearn.model_selection import train_test_split
 
# Split the data into training and testing sets
# 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了
train_features, test_features, train_labels, test_labels = train_test_split(features_array,labels, test_size = 0.25,random_state = 2)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

print("划分数据集和测试集完成==============================================")


X = train_features
y = train_labels






#==========================================================
#开始训练模型
#print("fdsfsd")
print("开始训练！")
n_folds = 10  # 设置交叉检验的次数

model = BayesianRidge()
#model = SVR(kernel='rbf')  # 建立支持向量机回归模型对象
#model = sklearn.svm.SVC(C=1.0,kernel='rbf', degree=3, gamma='auto',coef0=0.0,shrinking=True,probability=False,tol=0.001,cache_size=200, class_weight=None,verbose=False,max_iter=-1,decision_function_shape=None,random_state=None)


    
#scores = cross_val_score(model_lr, X, y, cv=n_folds,scoring = 'r2')  # 将每个回归模型导入交叉检验模型中做训练检验
#cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
model.fit(X, y)

Pre = model.predict(test_features)


model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表


print(test_labels)

pre = np.array(Pre)
print(Pre)

tmp_list = []  # 每个内循环的临时结果列表
for m in model_metrics_name:  # 循环每个指标对象
    score = m(test_labels, Pre)  # 计算每个回归指标结果
    
    strr = str(score)
    print(":%f" %score)
    #tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表











