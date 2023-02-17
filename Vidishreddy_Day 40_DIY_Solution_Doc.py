#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("TaxiFare.csv")
df.head()


# In[3]:


df= df.drop("unique_id",axis = 1)


# In[4]:


df


# In[5]:


df.isna().sum()


# In[6]:


df["date_time_of_pickup"] = pd.to_datetime(df["date_time_of_pickup"])
new_df = df.assign(hour = df["date_time_of_pickup"].dt.hour, 
                  dayOfTheMonth = df["date_time_of_pickup"].dt.day,
                  month = df["date_time_of_pickup"].dt.month, 
                  dayOfTheWeek = df["date_time_of_pickup"].dt.dayofweek)

# Remove date_time_of_pickup
new_df.drop("date_time_of_pickup", axis = 1, inplace = True)

new_df.head()


# In[7]:


import numpy as np


# In[8]:


def haversine_np(lon1, lat1, lon2, lat2):
  
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c # 6367 is radius of earth in kilometers.
    return km


# In[9]:


new_df["distance"] = haversine_np(new_df["longitude_of_pickup"], new_df["latitude_of_pickup"],
                                   new_df["longitude_of_dropoff"], new_df["latitude_of_dropoff"])

new_df.head()


# In[10]:


new_df.describe().transpose()


# A. Amount < 2.5 as the minimum fare is $2.5

print(new_df["amount"].describe())
fullRaw = new_df[new_df["amount"] >= 2.5]
print(new_df["amount"].describe())


# In[11]:


# B. Trips with travel distance greater than or equal to 1, and less than 130Kms.

print(new_df["distance"].describe())
new_df = new_df[(new_df["distance"] >= 1) & (new_df["distance"] <= 130)]
print(new_df["distance"].describe())


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report,accuracy_score
from sklearn.metrics import roc_curve , auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[13]:


x = new_df.drop(["amount"], axis = 1).copy()
y = new_df["amount"].copy()


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=100) 


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[14]:



from sklearn.ensemble import RandomForestRegressor
M1 = RandomForestRegressor(random_state=123)
M1 = M1.fit(x_train,y_train)

varImpDf = pd.DataFrame()
varImpDf["Importance"] = M1.feature_importances_
varImpDf["Variable"] = x_train.columns
varImpDf.sort_values("Importance", ascending = False, inplace = True)

varImpDf.head()


# In[15]:


# Model Prediction on Testset

testPredDf = pd.DataFrame()

testPredDf["Prediction"] = M1.predict(x_test)

# Create a column to store actuals
testPredDf["Actual"] = y_test.values

# Validate if the above worked
testPredDf.head()


# In[16]:


# RMSE
print("RMSE",np.sqrt(np.mean((testPredDf["Actual"] - testPredDf["Prediction"])**2)))

# MAPE
print("MAPE",(np.mean(np.abs(((testPredDf["Actual"] - testPredDf["Prediction"])/testPredDf["Actual"]))))*100)


# In[17]:


#Q2. Problem Statement: Performance Measurements


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing  import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import codf1.shapenfusion_matrix
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
from sklearn import metrics
import seaborn as sns


# In[20]:


df1 = pd.read_csv("h1n1_vaccine_prediction.csv")
df1.head()


# In[21]:


df1.h1n1_vaccine.value_counts()


# In[22]:


df1.shape


# In[23]:


df1.isnull().sum()/len(df1)*100  


# In[24]:


len(df1)


# In[25]:


df1 = df1.drop("has_health_insur",axis = 1)
df1 = df1.dropna()
df1.shape


# In[27]:


purchased=df1[df1.h1n1_vaccine==0].h1n1_vaccine.count()
notpurchased=df1[df1.h1n1_vaccine==1].h1n1_vaccine.count()
plt.bar(0,purchased,label='no')
plt.bar(1,notpurchased,label='yes')
plt.xticks([])
plt.ylabel('Count')
plt.legend()
plt.show()


# In[28]:


df1.info()


# In[29]:


from sklearn import preprocessing
df1.age_bracket.unique() # we need to replce age bracate with labes
le = preprocessing.LabelEncoder()
df1['age_bracket'] = le.fit_transform(df1.age_bracket.values)
df1['age_bracket']


# In[30]:


df1.qualification = le.fit_transform(df1.qualification.values)
df1.qualification


# In[31]:


df1.race = le.fit_transform(df1.race.values)
df1.sex = le.fit_transform(df1.sex.values)
df1.income_level = le.fit_transform(df1.income_level.values)
df1.marital_status = le.fit_transform(df1.marital_status.values)
df1.housing_status = le.fit_transform(df1.housing_status.values)
df1.employment = le.fit_transform(df1.employment.values)
df1.census_msa = le.fit_transform(df1.census_msa.values)


df1 # all column converted to numeric data


# In[32]:


x= df1.drop("h1n1_vaccine",axis = 1) #independent feature
y= df1['h1n1_vaccine'] # dependent feature 


# In[33]:


#before smot
x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.20)
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(x_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(x_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 
print()
score1 = metrics.accuracy_score(y_test, y_pred)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# In[35]:


#Over sampling the dataset
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
x_smot, y_smot = oversample.fit_resample(x, y)


# In[36]:


# after smot all class are equale.
purchased=y_smot[y_smot==0].count()
notpurchased=y_smot[y_smot==1].count()
plt.bar(0,purchased,label='no')
plt.bar(1,notpurchased,label='yes')
plt.xticks([])
plt.ylabel('Count')
plt.legend()
plt.show()


# In[ ]:




