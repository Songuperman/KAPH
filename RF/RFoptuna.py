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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
transfer = MinMaxScaler()
#transfer = StandardScaler()
x = transfer.fit_transform(x)
y = data.loc[:, 'target']
x = np.array(x)
y = np.array(y)

################################################CAT############################################
def optuna_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 500, 2500, 10)  # 整数型，(参数名称，下界，上界，步长)
    max_depth = trial.suggest_int("max_depth", 1, 10, 1)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5, 1)

    from sklearn.ensemble import RandomForestClassifier
    reg = RandomForestClassifier(n_estimators=n_estimators, random_state=22
                                , max_depth=max_depth
                                , verbose=False
                                , n_jobs=-1
                                , min_samples_split=min_samples_split
                                )

    from sklearn.model_selection import cross_validate, KFold
    from sklearn.metrics import r2_score, mean_squared_error, make_scorer
    scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro']
    cv = KFold(n_splits=10, shuffle=True, random_state=22)
    validation_loss = cross_validate(reg, x, y
                                     , scoring= scoring#"neg_root_mean_squared_error"
                                     , cv=cv  # 交叉验证模式
                                     , verbose=False  # 是否打印进程
                                     , n_jobs=-1  # 线程数
                                     , error_score='raise'
                                     , return_train_score= True
                                     )
    # 最终输出RMSE
    return  np.mean(abs(validation_loss['test_f1_macro']))

def optimizer_optuna(n_trials, algo):
    # 定义使用TPE或者GP
    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=100)  # 默认最开始有10个观测值，每一次计算采集函数随机抽取24组参数组合
    elif algo == "GP":
        from optuna.integration import SkoptSampler  # skilearn——optimize
        import skopt
        algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',  # 选择高斯过程
                                          'n_initial_points': 10,  # 初始观测点10个
                                          'acq_func': 'EI'}  # 选择的采集函数为EI，期望增量
                            )

    # 实际优化过程，首先实例化优化器
    study = optuna.create_study(sampler=algo  # 要使用的具体算法 sampler对样本进行抽样
                                , direction="maximize"  # 优化的方向，可以填写minimize或maximize
                                )
    # 开始优化，n_trials为允许的最大迭代次数
    # 由于参数空间已经在目标函数中定义好，因此不需要输入参数空间
    study.optimize(optuna_objective  # 目标函数
                   , n_trials=n_trials  # 最大迭代次数（包括最初的观测值的）
                   , show_progress_bar=True  # 要不要展示进度条呀？
                   )

    # 可直接从优化好的对象study中调用优化的结果
    # 打印最佳参数与最佳损失值
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")

    return study.best_trial.params, study.best_trial.values


import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

# best_params, best_score = optimizer_optuna(10, "GP")  # 默认打印迭代过程
# optuna.logging.set_verbosity(optuna.logging.ERROR)  # 关闭自动打印的info，只显示进度条
# optuna.logging.set_verbosity(optuna.logging.INFO)
best_params, best_score = optimizer_optuna(100, "TPE")
optuna.logging.set_verbosity(optuna.logging.ERROR)
# best_params, best_score = optimizer_optuna(300, "GP")


# best
# params: {'n_estimators': 1120, 'max_depth': 3, 'min_samples_split': 5}
#
# best
# score: [0.6345797736296774]