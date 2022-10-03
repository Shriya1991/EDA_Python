#!/usr/bin/env python
# coding: utf-8

# ## EDA Step by Step

# ##### The objective here is to perform an exploratory analysis on 'Algerian_forest_fires_dataset_UPDATE' data to analyse distribution of individual variables including target (Classes) & independent variables (Univariate analysis), study relationship between the target 'Classes' & various independent variables (Xis) (Bivariate Analysis) and relationship among Xis (Multivariate Analysis) inorder to arrive at meaningful insights on the data  

# In[1]:


#Import/load libraries
import pandas as pd
import numpy as np


# In[2]:


#Read relevant data
df = pd.read_csv (r'C:\Users\balan\python_assignments\EDA\Algerian_forest_fires_dataset_UPDATE.csv', header=[0], skiprows=1)
#print 1st few rows
df.loc[121:125]


# In[3]:


#remove unwanted rows
df=df.drop([122, 123,124])


# In[4]:


#Add region variable (as the csv loaded had 2 sections of data covering 2 regions, after merging the data, region variable will help to differentiate between the two region observations)
df['index_val'] = df.index
df['region']=np.where(df['index_val'] <=121, "Region1", "Region2")
df = df.drop("index_val", axis = 1)


# In[5]:


#Print dimention - Number of rows & columns 
df.shape


# In[6]:


#check if unwanted rows have been removed
print(df.head(125))


# In[7]:


#print null counts 
df.info()


# In[8]:


#All variables are of type obejct whereas most of the columns have numeric data. To get summary variable type should be numeric- change it numeric. 
cols = [i for i in df.columns if i not in["Classes  " , "region","month", "year"]]
print(cols)
for col in cols:
    df[col]=pd.to_numeric(df[col])


# In[9]:


#check if datatype has changed or not
df.info()


# In[10]:


#Observed spaces in variable names & values (classes) and could be case with other columns 
#remove space from variable names 
print("Before removig spaces :",df.columns)#before
df.columns = df.columns.str.replace(' ','')  
print("After removig spaces :",df.columns)#after
# remove special character from onject variables
cols = [i for i in df.columns if i in['month', 'year',"Classes" , "region"]]
print(cols)
for col in cols:
    df[col] = df[col].str.strip()


# In[11]:


#Print unique values for categorical variables
for col in cols:
    print("Unique values for", col , ":",df[col].unique())


# In[12]:


# Display summary statistics for numeric variables from data
df.describe()


# In[13]:


# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype =="O"]

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# ### Feature Information - 
# 1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)
# Weather data observations
# 2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 3. RH : Relative Humidity in %: 21 to 90
# 4. Ws :Wind speed in km/h: 6 to 29
# 5. Rain: total day in mm: 0 to 16.8
# FWI Components
# 6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# 7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# 8. Drought Code (DC) index from the FWI system: 7 to 220.4
# 9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# 10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
# 11. Fire Weather Index (FWI) Index: 0 to 31.1
# 12. Classes: two classes, namely Fire and not Fire

# In[14]:


#Print frequency distribution for categorical variables (in proportions)
for i in categorical_features:
    print('---------------------------')
    print("Distribition of:",i)
    print(df[i].value_counts(normalize=True) * 100) 
    print('---------------------------')


# ##### Observations - Target is binary in nature and has ~57% falling under fire category. The other "not fire' class has relatively lower propotion. The other categorical variable has 50-50 distribution, so each region has equal sample size

# ## Univariate Analaysis  
# ##### The purpose of univariate analysis is to understand how is the underlying distributed. It involes analysing every variable present in the data individually one at a time

# In[15]:


#Pictorial representation of data 
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,15))
plt.suptitle('Distribution of numeric variables', fontsize =20,    fontweight ='bold', alpha =0.8, y=1.)
sns.set_palette("pastel")
for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df[numeric_features[i]],shade=True, color='r')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# **Observations**
# * These plots show how data for each variable has distributed across its mean value. 
# * DMC, DC, ISI, BUI, FWI Rain are positively skewed distribution whereas FFMC, RH show negatively skewed data.
# * Temperature , VS somewhat look normally distributed

# In[16]:


# categorical columns
plt.figure(figsize=(20, 15))
plt.suptitle('Distribbution of Categorical Variables', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(categorical_features)):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=df[categorical_features[i]])
    plt.xlabel(categorical_features[i])
    plt.xticks(rotation=45)
    plt.tight_layout()


# **Observations**
# 
# * Each month has more or less same representation in the data 
# * Both the regions have exactly equally split
# * Year has only one value so 100% data would fall under one value
# * Classes which is the target variable shows some differences across 2 levels, fire occurences are higher in number compared to no fire instances

# In[17]:


#outlier detection through box plots -
plt.figure(figsize=(20, 15))
sns.boxplot(data=df)


# **Observation**
# * Wind speed (ws) has outliers on both the lower & upper ends
# * Rain is showing outliers on the higher side only 
# * Duff Moisture Code (DMC), Buildup index (BUI) has outliers on the upper ends and seems to have significant in number 
# * Fine fuel Moiture Code (FFMC) has outliers only on the lower end 
# * Though Temperature, Fire Weather Index (FWI), Initial spread index(ISI) show outliers but they are small in numbers
# * Relative Humidity(RH) has not apparent outliers

# In[18]:


plt.figure(figsize=(10,15))
plt.suptitle('Boxplot to detect outliers in numeric Variables', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(numeric_features)):
    plt.subplot(6, 2, i+1)
    sns.boxplot(x = df[numeric_features[i]],
            y = df['Classes'])
    plt.xlabel(numeric_features[i])
    plt.xticks(rotation=45)
    plt.tight_layout()


# **Observation**
# * By group box plot here is trying to show how various variables (Xis) are distributed across 2 values of the target (Y) variable Classes. For example, Rain variable shows, that in case of fire, we see very low rain has been reported with few outliers, whereas when cases when no fires are observed, we see that higher rain in the area has been reported with high number of outliers suggesting high millimeters or inches of rain
# * Higher temperature range has been seen for target = fire suggesting that possible reasons for higher number of fire events could be high temperature
# 

# In[19]:


#check correlation 
df[(list(df.columns)[1:])].corr()


# In[20]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), annot = True, fmt = '.2f')


# **Observations**
# If threshold be considered as > abs(0.75) - 
# * Duff Moisture Code (DMC) & Buildup Index (BUI) pair shows high positive correlation
# * Duff Moisture Code (DMC) & Drought Code (DC) pair shows high positive correlation
# * Drought Code (DC) & Buildup Index (BUI) pair shows high positive correlation
# * Buildup Index (BUI) & Fire Weather Index (FWI) pair shows high positive correlation
# * Fire Weather Index (FWI) with Duff Moisture Code (DMC), Initial Spread Index (ISI) & Buildup Index (BUI) show high positive correlation
# * We do not observe any strong correlation on the negative ends 
# 
# We can not measure corrlation between target & numeric variables as their scales are different.(Y is categorical & X is continuous)
# In case of categorical though, we can use chi-sqr test of independence to test if 2 Xis are associated or if there is any association between target and categorical Xis. 
# 
# * A chi-squared test (also chi-square or χ2 test) is a statistical hypothesis test that shows a relationship between two categorical variables.**
# * Here we test independence of Categorical columns with Target column i.e Classes**
# 
# 
# 

# In[21]:


from scipy.stats import chi2_contingency
chi2_test = []
for feature in categorical_features:
    if chi2_contingency(pd.crosstab(df['Classes'], df[feature]))[1] < 0.05:
        chi2_test.append('Reject Null Hypothesis')
    else:
        chi2_test.append('Fail to Reject Null Hypothesis')
result = pd.DataFrame(data=[categorical_features, chi2_test]).T
result.columns = ['Column', 'Hypothesis Result']
result


# **Observation**
# * Chi-sqr test assumes the null hypothesis as there is no association between the target & the independent variable. Because the P value is lower than the alpha(los) i.e maximum risk of error , we can reject the null hypothesis almost in case of independent variables, except for year which is not a variable to be considered

# In[22]:


#-check for missing counts variable wise
df.isnull().sum()


# In[23]:


#---scatter plot to analyse the relation between the target & other X-variables 
plt.figure(figsize=(10,15))
plt.suptitle('Categorical Scatter Plot', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(numeric_features)):
    sns.catplot(data=df,x =  f'{numeric_features[i]}',y = 'Classes')


# In[24]:


#summary by target
for i in range(0, len(numeric_features)):
    print('---------------------------')
    print(df[[f'{numeric_features[i]}', 'Classes']].groupby('Classes').describe())
    print('---------------------------')


# **Observation (based on catplots and target wise Xis summary)**
# * In case of event of fire, we see Temperature values going up till ~42 whereas in case non-fire, not that high temperature values have been reported. Though just looking at the terperature value range for the two classes will not tell us if temperature information helps predict the 'fire event' probability, We can certainly draw some conclusions based on Weight of evidence & IV value 
# * Higher Relative Humidity is observed in case of non-fire events vs lower humidity or rather dry weather reported during Fire events 
# * Wind speed does not show a very stark difference fire or non-fire event 
# * Non-fire instances observed to have higher amount of rainfall, whereas Fire instances seem to be happening cases where low rainfall has been measured. The trend however does not seem to be very strong
# * In case of non-fire events, we clearly see lower range of Fine Fuel Moisture Code (FFMC) (29-82), whereas in cases of fire events, higher range of Fine Fuel Moisture Code (FFMC)  has  been observed (80-96). A clear, distinct pattern can be seen emerging from the FFMC index
# * In case of DMC, DC, ISI, BUI, FWI observed to have higher range in case fire events vs non-fire events. The difference seen though visually apparent, may not prove to be statistically significant unless tested otherwise

# ##### To get an initial estimate of importance of various feature - calculate IV & WOE
# The weight of evidence tells the predictive power of an independent variable in relation to the dependent variable. Since it evolved from credit scoring world, it is generally described as a measure of the separation of good and bad customers. "Bad Customers" refers to the customers who defaulted on a loan. and "Good Customers" refers to the customers who paid back loan.This concept can be used to get some initial guesstimates about variable importance
# 

# In[25]:


#Binning of numeric data - creating equal size bins 
for i in range(0, len(numeric_features)):
    df[f'{"bin_"+ numeric_features[i]}'] = pd.qcut(df[numeric_features[i]], 5,duplicates='drop')
df.head()


# In[26]:


df.info()


# In[27]:


#Refresh categorical variable list
categorical_features = [feature for feature in df.columns if df[feature].dtype.name in ("object" ,"category")]
print(categorical_features)
#--derive binary target
df['target']=np.where(df['Classes'] =='fire',1,0)


# In[28]:


#woe & iv function (found online)
def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    dset = dset.sort_values(by='WoE')
    
    return dset, iv


# In[29]:


import warnings
warnings.filterwarnings('ignore')
IV = []

for col in categorical_features:
    if calculate_woe_iv(df, col, 'target') [1] <= 0.02:
        IV.append('Useless for prediction') 
    elif calculate_woe_iv(df, col, 'target') [1] > 0.02 and calculate_woe_iv(df, col, 'target') [1] <= 0.1:
        IV.append('Weak Predictor')
    elif calculate_woe_iv(df, col, 'target') [1] > 0.1 and calculate_woe_iv(df, col, 'target') [1] <= 0.3:
        IV.append('Medium Predictor')
    elif calculate_woe_iv(df, col, 'target') [1] > 0.3 and calculate_woe_iv(df, col, 'target') [1] <= 0.5:
        IV.append('Strong Predictor')
    else:
        IV.append('Too good to be true')

result = pd.DataFrame(data=[categorical_features, IV]).T
result.columns = ['Column', 'Predictive Power']
result           
            


# **Observation**
# * Region, Wind Speed, FFMC index, ISI can be prove to be medium to strong predictors compared to remaining predictors from the list in order to predict fire event probability. 
# * Though as per IV rules, many are getting clubbed under 'Too good to be true' or 'Useless for prediction', This is to be used for directional purpose only and further deep dive analysis to be done to rule out varibles
