import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings( "ignore" )

data=pd.read_csv("/Users/jyotsanasharma/Desktop/Capstone/capstone_test.csv")
cols=(data.columns).to_numpy()

# So there are null values in column Shadow_In_Midday
#Handling Missing values by replacing it with the mean
data['Shadow_In_Midday']=data['Shadow_In_Midday'].fillna(data['Shadow_In_Midday'].mean())


#Checking again the missing values in the dataset
print(data.isnull().sum())
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['Plant_Type']= label_encoder.fit_transform(data['Plant_Type']) 
print(data.head())

x = data.drop(columns=['Plant_Type'])
y = data[cols[12]]

from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    # cross validation - it is used for better validation of model
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

'''from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, x, y)'''

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model = RandomForestClassifier()
classify(model, x, y)
