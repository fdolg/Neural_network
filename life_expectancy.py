import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer 
from sklearn.compose import ColumnTransformer 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam 
# 1 
dataset = pd.read_csv("life_expectancy.csv")

#2 Printing first entries 
print(dataset.head())
print(dataset.describe())

#3
dataset = dataset.drop(['Country'], axis=1)
#4
labels = dataset.iloc[:,-1]
#5 Getting features 
features = dataset.iloc[:,0:-1]

#Data preprocessing 
#6 Converting categorical columns into cuantitative variables 
features = pd.get_dummies(features)

#Spliting data 
features_train,  features_test, labels_train,labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 3)

#Standarization/ normalization 
#Getting numerical columns
numerical_features = features.select_dtypes(include = ['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder = 'passthrough')

#9 
features_train_scaled = ct.fit_transform(features_train)

#10
features_test_scaled = ct.fit_transform(features_test)
#11
my_model = Sequential()

#12 
input = InputLayer(input_shape = (features.shape[1],))

#13 
my_model.add(input)
#14 Adding Hidden layer 
my_model.add(Dense(64, activation = "relu"))
#15
my_model.add(Dense(1))

# 16 Printing model's summary 
print(my_model.summary())

### Initializing the optimize and compiling the model 
#17
opt = Adam(learning_rate = 0.01)

#18 
my_model.compile(loss= 'mse', metrics = ['mae'], optimizer = opt)

#19 
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 0)

#20 Evaluating 
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)

#21 
print(res_mse, res_mae)




