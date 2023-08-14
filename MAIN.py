from  catboost  import CatBoostRegressor
import lightgbm
import numpy as np
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import os
from sklearn.ensemble import RandomForestRegressor
root = 'kap_total.xlsx'
data = pd.read_excel(root)
li = list(data.columns)
li.remove("target")
print(li)
x = data.loc[:, li]
from sklearn.preprocessing import MinMaxScaler
transfer = MinMaxScaler()
x = transfer.fit_transform(x)
y = data.loc[:, 'target']
x = np.array(x)
y = np.array(y)

from sklearn.ensemble import RandomForestClassifier
##########################################                 CAT                   ############################################
# model = CatBoostClassifier(iterations=3200,
#                               depth= 1,
#                               learning_rate=0.759,
#                              border_count =136,
#                              l2_leaf_reg =9,
#                               random_seed = 22,
#                              subsample=1,
#                             loss_function = "CrossEntropy"
#                              )

############################################     light     ################################################
# model = lightgbm.LGBMClassifier(learning_rate=0.636,
#                               n_estimators=1790,
#                               max_depth=1,
#                               min_child_samples=10,
#                               random_state=22,num_leaves= 49
#                               )

######################################### RF ##################################
model = RandomForestClassifier(n_estimators= 1780, random_state=22
                             , max_depth= 10
                             , min_samples_split= 5
                             )


######################################    DT    ##################################
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(random_state=22
#                              , max_depth=5
#                              , min_samples_split=5
#                              )
# model.fit(x,y)

from sklearn.model_selection import cross_validate, KFold
scoring = ['precision_macro', 'recall_macro','accuracy','f1_macro','roc_auc']
cv = KFold(n_splits=10, shuffle=True, random_state=22)
output = cross_validate(model, x, y
                                 , scoring= scoring
                                 , cv=cv
                                 , verbose=False
                                 , n_jobs=-1
                                 , error_score='raise'
                                 , return_train_score=True
                                 )

print('mean_train_recall_macro',np.mean(output['train_recall_macro']))
print('mean_train_precision_macro',np.mean(output['train_precision_macro']))
print('mean_train_accuracy',np.mean(output['train_accuracy']))
print('mean_train_f1',np.mean(output['train_f1_macro']))
print('mean_test_recall_macro',np.mean(output['test_recall_macro']))
print('mean_test_precision_macro',np.mean(output['test_precision_macro']))
print('mean_test_accuracy',np.mean(output['test_accuracy']))
print('mean_test_f1',np.mean(output['test_f1_macro']))
print('mean_test_roc_auc',np.mean(output['test_roc_auc']))

import shap
explainer  = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)
x = pd.DataFrame(x)
x.columns =['Knowledge', 'Attitude', 'Practice']

labels_index= {'Unhealth': 0, 'Health': 1}


##############################################           SUMMARY               ###########################
# import matplotlib
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
#############################################     All DATA BAR ###############################################
# shap.summary_plot(shap_values,x,
#                   class_names= list(labels_index.keys()),
#                   feature_names = x.columns,
#                   plot_type="bar",plot_size=(5,5)
#                   )
############################################      All DATA Bee   ###############################################
# plt.title("output  =  Health1")
# shap.summary_plot(shap_values[1], x.values, feature_names = x.columns,plot_size=(8,8))
#
# # plt.title("output  =  Health")
# shap.summary_plot(shap_values[1], x.values, feature_names = x.columns,plot_type="layered_violin",plot_size=(8,8))
# plt.show()

# plt.title("output  =  Unhealth")
# shap.summary_plot(shap_values[0], x.values, feature_names = x.columns,plot_type="layered_violin",plot_size=(8,8))
################################################      Interaction    ###########################################

######################################################### K A ####################################################
# # plt.title("output  =  Health")

# shap.dependence_plot('Knowledge', shap_values[1], x ,interaction_index="Attitude")
# # plt.title("output  =  Unhealth")
# shap.dependence_plot('Knowledge', shap_values[0], x ,interaction_index="Attitude", title="output  =  Unhealth")

###################################################### K  P ###########################################################
# # plt.title("output  =  Health")

# shap.dependence_plot('Knowledge', shap_values[1], x ,interaction_index="Practice")
# # plt.title("output  =  Unhealth")
# shap.dependence_plot('Knowledge', shap_values[0], x ,interaction_index="Practice", title="output  =  Unhealth")

#######################################################  A   P###########################################################
# # plt.title("output  =  Health")

# shap.dependence_plot('Attitude', shap_values[1], x ,interaction_index="Practice")
# # plt.title("output  =  Unhealth")
# shap.dependence_plot('Attitude', shap_values[0], x ,interaction_index="Practice", title="output  =  Unhealth")
###########################################################  p ########################################################
# plt.title("output  =  Health")

# shap.dependence_plot('Practice', shap_values[1], x)
# plt.title("output  =  Unhealth")
# shap.dependence_plot('Attitude', shap_values[0], x ,interaction_index="Practice", title="output  =  Unhealth")
# plt.show()
####################################################   Local  ############################################################
###################################                  C vs V        ################################################
# root1 = 'D:\\variable_importance\ZKAP\kap_CV.xlsx'
root1 ='kap_CV.xlsx'
data1 = pd.read_excel(root1)
li = list(data1.columns)
li.remove("target")
print(li)
x1 = data1.loc[:, li]
from sklearn.preprocessing import MinMaxScaler
transfer = MinMaxScaler()
x1 = transfer.fit_transform(x1)
y1 = data.loc[:, 'target']
x1 = np.array(x1)
y1 = np.array(y1)

explainer1  = shap.TreeExplainer(model)
shap_values1 = explainer1.shap_values(x1)
x1 = pd.DataFrame(x1)
x1.columns =['Knowledge', 'Attitude', 'Practice']
labels_index1= {'Unhealth': 0, 'Health': 1}
plt.title("City")
# shap.summary_plot(shap_values1[1][0:579], x1.values[0:579],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
plt.title("Village ")
# shap.summary_plot(shap_values1[1][579:], x1.values[579:],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
# shap.summary_plot(shap_values1[0][579:], x1.values[579:],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
plt.show()

#####################################################             Gender             ##############################################
root2 = 'kap_gender.xlsx'
data2 = pd.read_excel(root2)
li = list(data2.columns)
li.remove("target")
print(li)
x2 = data2.loc[:, li]
from sklearn.preprocessing import MinMaxScaler
transfer = MinMaxScaler()
x2 = transfer.fit_transform(x2)
y2= data.loc[:, 'target']
x2 = np.array(x2)
y2 = np.array(y2)

explainer2  = shap.TreeExplainer(model)
shap_values2 = explainer2.shap_values(x2)
x2 = pd.DataFrame(x2)
x2.columns =['Knowledge', 'Attitude', 'Practice']
labels_index2= {'Unhealth': 0, 'Health': 1}
plt.title("Male")
# shap.summary_plot(shap_values2[1][0:673], x2.values[0:673],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
# shap.summary_plot(shap_values2[0][0:673], x2.values[0:673],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
plt.title("Female")
# shap.summary_plot(shap_values2[1][673:], x2.values[673:],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
# plt.title("Female (output = Unhealth)")
# shap.summary_plot(shap_values2[0][673:], x2.values[673:],
#                   feature_names = x.columns,
#                   plot_type="layered_violin",
#                   plot_size=(8,8))
plt.show()