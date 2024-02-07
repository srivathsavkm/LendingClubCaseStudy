#!/usr/bin/env python
# coding: utf-8

# In[442]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[443]:


# Data loading

loan = pd.read_csv('loan.csv')
loan.head()


# In[444]:


loan.shape


# In[445]:


# Print all columns
for col in loan.columns:
    print(col)


# # Data Cleaning

# In[446]:


#Check all columns that are null
print(loan.isnull().all(axis=0).sum())


# In[447]:


# Percentage of null values in columns
for col in loan.columns:
    print(col, (loan[col].isnull().sum()/ len(loan))*100)


# ### 1. Drop unwanted columns

# In[448]:


# Drop all columns which doesn't have any values

arr = []
for col in loan.columns:
    if loan[col].isnull().sum() == len(loan):
        arr.append(col)
loan.drop(labels=arr, axis = 1, inplace=True)


# In[449]:


loan.shape


# ### 2. Data impute

# In[450]:


# emp_length
loan.emp_length.fillna('-1', inplace=True)
loan['emp_length'] = loan.emp_length.str.extract('(\d+)')
loan.emp_length


# In[451]:


# pub_rec_bankruptcies 
loan.pub_rec_bankruptcies.isnull().sum()
loan.pub_rec_bankruptcies.fillna(-1, inplace=True)
loan.pub_rec_bankruptcies


# In[452]:


# int_rate 
loan['int_rate'] = loan.int_rate.str.rstrip('%')
loan['int_rate']


# In[453]:


# Convert numeric columns to number fields
cols1 = ['loan_amnt','funded_amnt','int_rate','funded_amnt_inv','installment','annual_inc','dti','emp_length','total_pymnt']
loan[cols1] = loan[cols1].apply(pd.to_numeric)


# In[454]:


# Percentage loan_status review
round((loan.loan_status.value_counts()*100)/len(loan), 2)


# In[455]:


# loan purpose
round((loan.purpose.value_counts()*100)/len(loan), 2)


# ### 3. Derived columns

# In[456]:


# Get year and month columns based on issue date
loan.issue_d = pd.to_datetime(loan.issue_d, format='%b-%y')
loan['year'] = loan.issue_d.dt.year
loan['month'] = loan.issue_d.dt.month


# In[457]:


# Get loan amount categories from loan_amnt
loan['loan_amnt_cats'] = pd.cut(loan['loan_amnt'], [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000], labels=['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000', '25000-30000', '35000+'])


# In[458]:


# Get annual income categories from annual_inc
loan['annual_inc_cats'] = pd.cut(loan['annual_inc'], [0, 20000, 40000, 60000, 80000,1000000], labels=['0-20000', '20000-40000', '40000-60000', '60000-80000', '80000 +'])


# In[459]:


# Get interest rate categories from int_rate
loan['int_rate_cats'] = pd.cut(loan['int_rate'], [0, 5, 10, 15, 20, 25], labels=['0-5', '5-10', '10-15', '15-20', '20+'])


# In[460]:


# Get defaulted column from loan_status
loan['IsDefaulted'] =  loan['loan_status'].apply(lambda x: 1 if (x == 'Charged Off') else 0)


# In[540]:


# Get dti categories from dti
loan['dti_cats'] = pd.cut(loan['dti'], [-1, 5, 10, 15, 20, 25, 30], labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25+'])


# # Univariate Analysis

# In[569]:


#Defaulter vs Non-Defaulter
#Defaulter = 1 and Non-Defaulter = 0
# we can see the ratio of defaulters vs non-defaulters
plt.figure(figsize=(8,6))
loan['IsDefaulted'].value_counts().plot.bar()


# In[461]:


# loan_amnt analysis

loan['loan_amnt'].describe()
plt.boxplot(loan['loan_amnt'])
plt.show


# In[462]:


# total_pymnt analysis

loan.total_pymnt.describe()
plt.boxplot(loan.total_pymnt)
plt.show()


# In[463]:


# annual_inc analysis

loan = loan[loan['annual_inc'] < loan['annual_inc'].quantile(0.99)]
loan.annual_inc.describe()


# In[464]:


plt.boxplot(loan.annual_inc)
plt.show()


# In[465]:


# int_rate analysis

loan.int_rate.describe()


# In[466]:


plt.boxplot(loan.int_rate)
plt.show()


# # Data visualization

# In[467]:


# Set the figure size and background color
plt.figure(figsize=(10, 6), facecolor='w')

# Create a count plot using seaborn
ax = sns.countplot(x="term", data=loan, hue='loan_status', palette='viridis')

# Set labels and title
ax.set_title('Loan Paying Term', fontsize=14, color='w')
ax.set_xlabel('Loan Repayment Term', fontsize=14, color='w')
ax.set_ylabel('Loan Application Count', fontsize=14, color='w')

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

# Display exact count on top of bars without a loop
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge', fontsize=10, color='black', weight='bold')
ax.bar_label(ax.containers[1], fmt='%d', label_type='edge', fontsize=10, color='black', weight='bold')

# Show the plot
plt.show()


# In[576]:


def plot_graph(col):
    arr = []
    for val in loan[col].unique():
        arr.append(len(loan[(loan[col] == val) & (loan['IsDefaulted'] == 1)]) / len(loan[(loan[col] == val)]))
    # Hardcoded values
    categories = loan[col].unique()
    values = arr

    # Create a bar graph
    colors = ['#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    plt.bar(categories, values, color=colors)
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90, ha='right')  # You can adjust the rotation angle as needed
    # Add labels and title
    plt.xlabel(col)
    plt.ylabel('Defaulted(%)')
    plt.title('bar graph')

    # Show the plot
    plt.show()


# In[577]:


def BarChartBivariate(col, hue, data):
    plt.figure(figsize=(8,6))
    sns.barplot(x="IsDefaulted", y=col, hue=hue, data=data)


# In[578]:


# term: 60 Months are defaulted more compared to 36 Months
plot_graph('term')


# In[579]:


# home_ownership: Not a good indicator for defaulting
plot_graph('home_ownership')


# In[580]:


# Purpose: By looking at the below graph, small_business defaulted the most
plot_graph('purpose')


# In[581]:


# verification_status: verified users have defaulted the most
plot_graph('verification_status')


# In[582]:


# emp_length: not a good indicator for loan dafault
plot_graph('emp_length')


# In[583]:


# pub_rec_bankruptcies: have higher default rates where bankruptcies = 2
plot_graph('pub_rec_bankruptcies')


# In[584]:


# grade: grade G defaulted the most and 2nd highest is F
plot_graph('grade')


# In[585]:


# sub_grade: sub-grade F5 defaulted the most
plot_graph('sub_grade')


# In[586]:


# loan_amnt_cats: 35000 and above defaulted the most
plot_graph('loan_amnt_cats')


# In[587]:


# annual_inc_cats: 0-20000 salary defaulted the most
plot_graph('annual_inc_cats')


# In[588]:


# int_rate_cats: 20% and above interest rates defaulted the most
plot_graph('int_rate_cats')


# In[589]:


# year: based on the below graph, in the year 2007 there were more number of defaulters
plot_graph('year')


# In[590]:


# dti_cats: based on the below graph, 20-25% dti have defaulted the most
plot_graph('dti_cats')


# In[591]:


# purpose counts
round((loan.purpose.value_counts()*100)/len(loan), 2)


# In[575]:


color = ['#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

plt.figure(figsize=(14,8),facecolor='#4363d8')
sns.set_style("dark")
ax = sns.countplot(y="purpose",data=loan,hue='loan_status',palette='GnBu_d')
ax.set_title('Purpose Of Loan',fontsize=14,color='w')
ax.set_ylabel('Purpose Of Loan',fontsize=14,color = 'w')
ax.set_xlabel('Loan Application Count',fontsize=14,color = 'w')      
plt.show()

# debt_consolidation has been charged off the most

