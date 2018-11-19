import pandas as pd
import numpy as np

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import skew

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load The Data
TRAIN_FILE_PATH = 'C:/Users/ahlaw/Downloads/train.csv'
TEST_FILE_PATH = 'C:/Users/ahlaw/Downloads/test.csv'
training_data = pd.read_csv(TRAIN_FILE_PATH)
test_data = pd.read_csv(TEST_FILE_PATH)

training_rows = training_data.shape[0]
test_data['SalePrice'] = 0
all_data = pd.concat([training_data, test_data]).reset_index(drop = True)

#remove outliers, 2 rows have very large GrLivArea but very small SalePrice
all_data = all_data.drop(all_data[(all_data['GrLivArea']>4000) & (all_data['SalePrice']<300000) & (all_data['SalePrice'] != 0)].index)
training_rows = training_rows -  2

#Handle missing values

#We impute values of Categorical features with their mode and numerical features with their median unless the missing values
#have some special meaning or can be handled by observing the other features

#We impute below categorical features with their mode
misc_features = ['Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
for feature in misc_features:
    all_data[feature].fillna(all_data[feature].mode()[0], inplace = True)

#Missing Values of PoolQC indicate the absence of a pool. We can check if any values are missing even with a pool.
# all_data[(all_data['PoolArea'] != 0) & (all_data['PoolQC'].isnull())]
#After executing above statement we can see 3 rows have PoolQC missing with a pool. We can estimate PoolQC by the 
#OverallQual
all_data.loc[2420, 'PoolQC'] = 'Fa'
all_data.loc[2504, 'PoolQC'] = 'Gd'
all_data.loc[2600, 'PoolQC'] = 'Fa'

#Missing values of GarageArea indicate no garage
all_data['GarageArea'].fillna(0, inplace = True)

#For rows where there is a garage we simply impute Garage categorical features with their mode and numerical
#features with their median
garage_feats = ['GarageQual', 'GarageCond', 'GarageFinish']
for feat in garage_feats:
    all_data.loc[(all_data['GarageArea'] != 0) & (all_data[feat].isnull()), feat] = all_data[feat].mode()[0] 

all_data.loc[(all_data['GarageArea'] != 0) &
             (all_data['GarageYrBlt'].isnull()), 'GarageYrBlt'] = all_data['GarageYrBlt'].median()
all_data.loc[(all_data['GarageArea'] != 0) &
             (all_data['GarageCars'].isnull()), 'GarageCars'] = all_data['GarageCars'].median()

#Missing TotalBsmtSF indicates absence of a Basement
all_data['TotalBsmtSF'].fillna(0, inplace = True)

#Some rows have missing Basement features even with a basement, we can check this by executing below statement
# all_data[(all_data['TotalBsmtSF'] != 0) & (all_data['BsmtQual'].isnull())]
#For BsmtQual we can get an estimate using the size of the basement and BsmtCond
all_data.loc[2217, 'BsmtQual'] = 'Fa'
all_data.loc[2218, 'BsmtQual'] = 'TA'

#Again, we can check for rows with a basement but no BsmtCond Values using below statement
# all_data[(all_data['TotalBsmtSF'] != 0) & (all_data['BsmtCond'].isnull())]
#We can estimate BsmtCond using  BsmtFinType1 and BsmtFinType2
all_data.loc[2040, 'BsmtCond'] = 'Gd'
all_data.loc[2185, 'BsmtCond'] = 'Fa'
all_data.loc[2524, 'BsmtCond'] = 'TA'

#For the rows having a basement and no BsmtExposure value, we can observe they are unfinished and its safe to assume
#they wont have any exposure
all_data.loc[(all_data['TotalBsmtSF'] != 0) &
             (all_data['BsmtExposure'].isnull()), 'BsmtExposure'] = 'No'

#Use the below statement to check missing BsmtFinType2 values in houses with basements
# all_data[(all_data['TotalBsmtSF'] != 0) & (all_data['BsmtFinType2'].isnull())]
#The below house has high overallqual and moderate price, so, we can get a good estimate
all_data.loc[332, 'BsmtFinType2'] = 'ALQ'

#Missing MassVnrArea indicate no Veneer
all_data['MasVnrArea'].fillna(0, inplace = True)

#We can find houses with veneer but missing MasVnrType using below statement
# all_data[(all_data['MasVnrArea'] != 0) & (all_data['MasVnrType'].isnull())]
#This house has very low OverallQual so its safe to assume it wont have any MasVnrType
all_data.loc[2610, 'MasVnrType'] = 'None'

#Houses in same MSSubclass should have same MSZoning 
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#For below categorical features missing values indicate their absence
feats = ['PoolQC', 'MiscFeature', 'Alley', 'FireplaceQu', 'Fence', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType',
        'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'MasVnrType']
for feat in feats:
    all_data[feat].fillna('None', inplace = True)
    
#Lotfrontage should be similar for houses in the  same neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#For below numerical features missing vaLues indicate their absence
feats = ['GarageYrBlt', 'BsmtHalfBath', 'BsmtFullBath', 'GarageCars', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']
for feat in feats:
    all_data[feat].fillna(0, inplace = True)
    
#This takes care of any rows with wrong GarageYrBlt
all_data.loc[all_data['GarageYrBlt'] > all_data['YrSold'], 'GarageYrBlt'] = all_data['YrSold']

#MSSubClass is stores as a numerical feature but is actually categorical, so, we convert it
all_data['MSSubClass'] = all_data['MSSubClass'].astype('str')

#linear models behave well with centred data, so, we will try to centre some of the skewed features
numeric_feats = list(all_data.select_dtypes(include = np.number).columns)
numeric_feats = [e for e in numeric_feats if e not in ('Id', 'SalePrice')]
skewness = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skewness = skewness[(skewness) > 0.75]
skewed_feats = list(high_skewness.index)
for feat in skewed_feats:   
    all_data[feat]= boxcox1p(all_data[feat], boxcox_normmax(all_data[feat]+1))

#Lets centre SalePrice as it is also skewed    
all_data["SalePrice"] = np.log1p(all_data["SalePrice"])

#Lets create new features from pre existing features
all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5*all_data['HalfBath']) + 
                               all_data['BsmtFullBath'] + (0.5*all_data['BsmtHalfBath']))

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])


all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data['Age'] = all_data['YrSold'] -  all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']

#Utilties and Street features have very less variance and donot provide any valuable information to our model
#YearRemodAdd and Yearbuilt are actually categorical features but have high variance thus making one hot encoding
#them impossible, so, create Age and RemodAge features in their stead.
all_data = all_data.drop(['Utilities', 'Street', 'YearRemodAdd', 'YearBuilt'], axis=1)

#One Hot encoding the categorical features
final_data = pd.get_dummies(all_data).reset_index(drop=True)

#Lets remove any features which have low variance
variance = final_data.var().sort_values()
variance = variance[variance < 0.01]
low_variance_feats = list(variance.index)
final_data.drop(low_variance_feats, axis = 1, inplace = True)

#Separate the data into training and test sets
X = final_data[:training_rows]
y = X['SalePrice']
X = X.drop(['Id', 'SalePrice'], axis = 1)

test_data = final_data[training_rows:]
test_ids = test_data['Id']
test_data = test_data.drop(['Id', 'SalePrice'], axis = 1)

#Initiatlize The models
#For the model parameters I used randomized and grid search with crossvalidation

ridge = make_pipeline(RobustScaler(), 
                      Ridge(alpha = 23.7))

lasso = make_pipeline(RobustScaler(),
                      Lasso(alpha = 0.0005))

elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNet(alpha=0.0007, l1_ratio=0.85))

lightgbm = make_pipeline(RobustScaler(),
                        LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11))

xgboost = make_pipeline(RobustScaler(),
                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 
                                     max_depth=3,min_child_weight=0 ,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective= 'reg:linear',nthread=4,
                                     scale_pos_weight=1,seed=27, 
                                     reg_alpha=0.00006))


#We use a stack model, which basically means the predications of our base regressors act as input to our 
#meta regressor with the SalePrice being predictor values
stack_gen = StackingCVRegressor(regressors=(ridge, elasticnet, lasso,
                                            xgboost, lightgbm), 
                               meta_regressor=xgboost,
                               use_features_in_secondary=True)

stackX = np.array(X)
stacky = np.array(y)

#Fit the models
elasticnet.fit(X, y)
lasso.fit(X, y)
ridge.fit(X, y)
xgboost.fit(X, y)
lightgbm.fit(X, y)
stack_gen.fit(stackX, stacky)

#We take a weighted average of our predictions, what this does adds some variance at the expense of losing a little bias
stack_preds = ((0.2*elasticnet.predict(test_data)) + (0.1*lasso.predict(test_data)) + (0.1*ridge.predict(test_data)) + 
               (0.2*xgboost.predict(test_data)) + (0.1*lightgbm.predict(test_data)) + (0.3*stack_gen.predict(test_data)))

#Create The Submission
sub = pd.DataFrame()
sub['Id'] = test_ids
sub['SalePrice'] = np.expm1(stack_preds)
sub.to_csv('submission.csv',index=False)
