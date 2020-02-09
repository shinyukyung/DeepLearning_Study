from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import shuffle
import tensorflow as tf
import utils
from train import ModelTraining
import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train   =   pd.read_csv("input/train.csv")
test    =   pd.read_csv("input/test.csv")
data    =   []


for f in train.columns:
    # role 정의
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # level 정의
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    # id를 제외한 모든 변수를 True로 초기화
    keep = True
    if f == 'id':
        keep = False
    
    # 데이터 유형 정의
    dtype = train[f].dtype
    
    # 변수에 대한 메타 데이터를 담는 사전 만들기
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)

meta[(meta.level == 'nominal') & (meta.keep)].index

pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()

v = meta[(meta.level == 'interval') & (meta.keep)].index
v = meta[(meta.level == "ordinal") & (meta.keep)].index
v = meta[(meta.level == "binary") & (meta.keep)].index

desired_apriori=0.10

# target 값에 따라 인덱스를 가져옵니다.
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# target 값에 따라 레코드의 원래 번호를 가져옵니다.
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# 언더샘플링 비율과 target이 0인 레코드 수를 계산합니다.
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# target이 0인 레코드를 무작위로 선택하여  언더샘플된 인덱스를 구합니다.
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# target이 1인 인덱스와 언더샘플된 인덱스를 합쳐 구성합니다.
idx_list = list(undersampled_idx) + list(idx_1)

# 언더샘플링 데이터프레임을 반환합니다.
train = train.loc[idx_list].reset_index(drop=True)

print(train.shape)
train["target"].value_counts()

vars_with_missing = []

for f in train.columns:
  missings = train[train[f] == -1][f].count()
  if missings > 0:
    vars_with_missing.append(f)
    missings_perc = missings / train.shape[0]
    
    print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))

# 누락 된 값이 너무 많은 변수는 제거
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # 메타 데이터 갱신
test.drop(vars_to_drop, inplace=True, axis=1)

# 평균 또는 모드로 대체하기
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()

v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    #print('Variable {} has {} distinct values'.format(f, dist_values))

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing은 Daniele Micci-Barreca의 논문에서와 같이 계산됩니다.
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : pd.Series 형태의 학습할 범주형 피처
    tst_series : pd.Series 형태의 테스트할 범주형 피처
    target : pd.Series 형태의 타겟 데이터
    min_samples_leaf (int) : 범주의 평균을 고려할 최소 샘플
    smoothing (int) : 범주 평균과 이전의 균형을 맞추기 위한 스무딩 효과 
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # 타겟 평균을 계산
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # 스무딩 계산
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # 모든 타겟 데이터에 평균 적용하기
    prior = target.mean()
    # 카운트가 클수록 full_avg가 적어집니다.
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # 학습, 테스트 데이터에 평균 적용
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge는 인덱스를 유지하지 않으므로 복원합니다.
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge는 인덱스를 유지하지 않으므로 복원합니다.
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)
test_id = test['id']


v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis=1, inplace=True)  # 원래의 컬럼 제거


train = pd.concat([train, interactions], axis=1)
selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id', 'target'], axis=1)) # id와 target 변수 없이 훈련하는 것이 적합하다.
f = np.vectorize(lambda x : not x) # 부울 배열 요소를 토글하는 함수
v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]


v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(test[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis=1, inplace=True)  # 원래의 컬럼 제거

test = pd.concat([test, interactions], axis=1)
selector = VarianceThreshold(threshold=.01)
selector.fit(test.drop(['id'], axis=1)) # id와 target 변수 없이 훈련하는 것이 적합하다.
f = np.vectorize(lambda x : not x) # 부울 배열 요소를 토글하는 함수
v = test.drop(['id'], axis=1).columns[f(selector.get_support())]

test = test.drop(['id'] , axis=1)

x_train, y_train, valid_x, valid_y = utils.split_valid_test_data(train)

##train
model = utils.build_neural_network(x_train)
traing_model = ModelTraining(model, x_train, y_train, valid_x, valid_y, 1, 50, 50 * 2, 0.001, 128)
traing_model.train()

##test
model   = utils.build_neural_network(test)
restorer=tf.train.Saver()

with tf.Session() as sess:
    restorer.restore(sess, "checkpoint/porto_pilsa.ckpt")
    feed={
        model.inputs: test, model.is_training: False}
    test_predict=sess.run(model.predicted,feed_dict=feed)


binarizer           =   Binarizer(0.5)
test_predict_result =   binarizer.fit_transform(test_predict)
test_predict_result =   test_predict_result.astype(np.int32)

test_id             =   test_id.copy()
evaluation          =   test_id.to_frame()
evaluation["target"]=   test_predict_result
evaluation.to_csv("input/evaluation_submission.csv",index=False)