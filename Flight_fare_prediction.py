#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


f_f_train = pd.read_excel('Data_Train.xlsx')
f_f_train.head()


# In[3]:


f_f_train.shape


# In[4]:


f_f_train.isnull().sum()


# In[5]:


f_f_train.dropna(inplace = True)


# In[6]:


f_f_train.info()


# In[7]:


f_f_train['Airline'].value_counts()


# In[8]:


f_f_train['Source'].value_counts()


# In[9]:


f_f_train['Destination'].value_counts()


# In[10]:


f_f_train['Additional_Info'].value_counts()


# #### Handling Date_of_Journey Column

# In[11]:


# Extracting day from the column

f_f_train['Journey_day'] = pd.to_datetime(f_f_train['Date_of_Journey'], format = '%d/%m/%Y').dt.day


# In[12]:


# Extracting month from the column

f_f_train['Journey_month'] = pd.to_datetime(f_f_train['Date_of_Journey'], format = '%d/%m/%Y').dt.month


# In[13]:


f_f_train.head()


# In[14]:


# Dropping the column 'Date_of_Journey'

f_f_train.drop('Date_of_Journey', axis = 1, inplace =True)


# In[15]:


f_f_train.head()


# #### Handling Dep_Time Column

# In[16]:


# Extracting hours and minutes from the column

f_f_train['Dep_hour'] = pd.to_datetime(f_f_train['Dep_Time']).dt.hour
f_f_train['Dep_minutes'] = pd.to_datetime(f_f_train['Dep_Time']).dt.minute


# In[17]:


# Dropping the column

f_f_train.drop('Dep_Time', axis = 1, inplace = True)


# In[18]:


f_f_train.head()


# #### Handling Arrival_Time Column

# In[19]:


# Extracting hours and minutes from the column

f_f_train['Arrival_hour'] = pd.to_datetime(f_f_train['Arrival_Time']).dt.hour

f_f_train['Arrival_minutes'] = pd.to_datetime(f_f_train['Arrival_Time']).dt.minute

f_f_train.drop('Arrival_Time', axis = 1, inplace = True)


# In[20]:


f_f_train.head()


# #### Handling Duration Column

# * If you will look at the duration column, most entries are in term of hours and minutes. There are also entries which has either only hours or only minutes present in it. So we will first level these entries so that each element will have a sub-element of hour and minute

# In[21]:


duration1 = list(f_f_train['Duration'])


# In[22]:


# Added 0 hours or 0 minutes in the elements wherever necessary

def add_hours_min(duration):
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if 'h' in duration[i]:
                duration[i] = duration[i].strip() + ' 0m'
            else:
                duration[i] = '0h ' + duration[i].strip()

add_hours_min(duration1)


# In[23]:


# Now we will seperate hours and minutes from duration list

duration_hours = []
duration_minutes = []
def sep_hour_min(duration):
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split('h')[0]))
        duration_minutes.append(int(duration[i].split('m')[0].split()[-1]))

sep_hour_min(duration1)


# In[24]:


# Now we will convert the list duration_hours and duration_minutes into columns

f_f_train['duration_hours'] = duration_hours
f_f_train['duration_minutes'] = duration_minutes


# In[25]:


f_f_train.drop('Duration', axis =1, inplace = True)


# In[26]:


f_f_train.head()


# #### Handling Airline Column

# * Airline is a categorical column and its of a cardinal type. So in this case we will use OneHotEncoding to convert it into a numerical column.

# In[27]:


Airline = f_f_train[['Airline']]


# In[28]:


Airline = pd.get_dummies(Airline, drop_first= True)
Airline.head()


# #### Handling Source Column

# * Source is again a cardinal categorical column. So we will use OneHotEncoding with the help of get_dummies.

# In[29]:


Source = f_f_train[['Source']]

Source = pd.get_dummies(Source, drop_first = True)

Source.head()


# #### Handling Destination Column

# * Destination is another cardinal categorical column. So we will use OneHotEncoding with the help of get_dummies.

# In[30]:


Destination = f_f_train[['Destination']]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# #### Handling Total_Stops Column

# * Total_Stops is an ordinal categorical column. It means that the elements have some order or importance in their respective column. In this particular case, the prices of flights with maximum number of stops will have maximum prices. for example: 4 stops > 3 stops > and so on. We will use Ordinal Encoding to deal with this particular categorical column. Basically assigning keys to the corresponding values.

# In[31]:


f_f_train["Total_Stops"].value_counts()


# In[32]:


f_f_train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
f_f_train.head()


# #### Handling Route and Additional_Info Column

# * We already have the substitute of Route column, i.e. Total_Stops column. While the Additional_Info column has maximum entries of 'no info'. We will drop both of these columns. 
# * We will also drop Airline, Source and Destination Column so that we can concatinate encoded columns afterwards.

# In[33]:


f_f_train.drop(['Airline','Source','Destination', 'Route', 'Additional_Info'], axis = 1, inplace = True)
f_f_train.head()


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[35]:


plt.figure(figsize = (18,18))
sns.heatmap(f_f_train.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[36]:


# Concatinating Encoded columns and creating a new dataframe (train_data)

train_data = pd.concat([ f_f_train, Airline, Source, Destination], axis = 1)


# In[37]:


train_data.head()


# In[38]:


train_data.shape


# In[39]:


train_data.columns


# ## Feature Selection and Model Creation
# 
# We have worked with both training and testing dataset. Its time to look for important features and create best model for predicting the flight prices. 

# In[40]:


# Selecting Dependent and Independent Features

X = train_data.drop('Price', axis = 1)
y = train_data['Price']


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[42]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)


# In[43]:


y_pred = rfr.predict(X_test)
y_pred


# In[44]:


sns.distplot(y_test-y_pred)
plt.show()


# In[45]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[46]:


print('R_square:',r2_score(y_test, y_pred))


# In[47]:


print('Mean_Absolute_Error: ',mean_absolute_error(y_test, y_pred))
print('Mean_Squared_Error: ',mean_squared_error(y_test, y_pred))
print('Root_Mean_Squared_Error: ',(mean_squared_error(y_test, y_pred))**(0.5))


# In[49]:


# Saving the model
import pickle

file = open('flight_price_rfr.pkl', 'wb')
pickle.dump(rfr, file)


# In[ ]:




