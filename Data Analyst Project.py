#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl 
import seaborn as sns

import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


import warnings
warnings.filterwarnings('ignore') 
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('data/DataAnalyst.csv')
df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isna().sum()


# In[9]:


df.replace("-1", np.nan)


# In[11]:


df.replace("-1", np.nan, inplace=True)
df.replace(-1, np.nan, inplace=True)
df.replace(-1.0, np.nan, inplace=True)


# In[12]:


df.isna().sum()


# In[13]:


df["Job Title"].value_counts().head(20)


# In[14]:


df["Job Title"] = df["Job Title"].str.replace("Sr. Data Analyst", "Senior Data Analyst")
df["Job Title"] = df["Job Title"].str.replace("Sr Data Analyst", "Senior Data Analyst")
df["Job Title"] = df["Job Title"].str.replace("Data Analyst Senior", "Senior Data Analyst")
df["Job Title"] = df["Job Title"].str.replace('Jr. Data Analyst','Junior Data Analyst')
df["Job Title"] = df["Job Title"].str.replace('Jr Data Analyst','Junior Data Analyst')
df["Job Title"] = df["Job Title"].str.replace('Data Analyst Junior','Junior Data Analyst')


# In[15]:


df["Job Title"].value_counts()[:20]


# In[16]:


df["Job Title"].isna().sum()


# In[17]:


df["Salary Estimate"].sample(5)


# In[18]:


df["Salary Estimate"].isna().sum()


# In[19]:


df[df["Salary Estimate"].isna()]


# In[20]:


df[df["Company Name"] == "Protingent\n4.4"]


# In[21]:


df["Job Description"][2123]


# In[22]:


df["Job Description"][2149]


# In[23]:


df.shape


# In[24]:


df.drop(2149, inplace=True)


# In[25]:


df["Salary Estimate"].isnull().sum()


# In[26]:


df["Salary Estimate"].sample(5)


# In[27]:


df["Salary Minimum"] = df["Salary Estimate"].str.lstrip("$").str[:3].str.replace("K", "").str.strip().astype("float")


# In[28]:


df["Salary Maximum"] = df['Salary Estimate'].str[6:10].str.replace('K','').str.lstrip('$').str.strip().astype('float')


# In[29]:


df["Salary Average"] = (df["Salary Maximum"] + df["Salary Minimum"]) / 2


# In[30]:


df[["Salary Estimate", "Salary Maximum", "Salary Minimum", "Salary Average"]].sample(5)


# In[31]:


df["Job Description"][0]


# In[32]:


df["Job Description"].isna().sum()


# In[33]:


df["python"] = df["Job Description"].str.contains("python", na=False, case=False)
df["python"].value_counts()


# In[34]:


df["SQL"] = df["Job Description"].str.contains("sql", na=False, case=False)
df["SQL"].value_counts()


# In[35]:


df["Excel"] = df["Job Description"].str.contains("excel", na=False, case=False)
df["Excel"].value_counts()


# In[36]:


df["Tableau"] = df["Job Description"].str.contains("tableau", na=False, case=False)
df["Tableau"].value_counts()


# In[37]:


df.Rating.sample(5)


# In[38]:


df.Rating.isna().sum()


# In[39]:


df["Company Name"].sample(10)


# In[40]:


df[["Company Name", "Rating"]].sample(10)


# In[41]:


df["Company Name"] = df["Company Name"].str.split("\n").str[0]
df["Company Name"].head()


# In[42]:


df["Company Name"].isna().sum()


# In[43]:


df[df["Company Name"].isna()]


# In[44]:


df.Industry.value_counts()


# In[45]:


df.Industry.isna().sum()


# In[46]:


df.Sector.value_counts()


# In[47]:


df.Sector.isna().sum()


# In[48]:


df.info()


# In[49]:


df.columns


# In[50]:


df_analyis = df[['Job Title', 'Company Name', 'Rating', 'Industry', 'Sector', 'Salary Minimum','Salary Maximum', 'Salary Average','python', 'SQL', 'Excel', 'Tableau']]
df_analyis.head()


# In[51]:


df_analyis.isna().sum()


# In[52]:


df_analyis.describe()


# In[53]:


fig = px.histogram(data_frame=df_analyis, x="Salary Minimum", title="Data Analyst Jobs - Minimum Salary", marginal="box", hover_data=df_analyis[["Job Title", "Company Name"]])
fig.show()


# In[54]:


fig = px.histogram(data_frame=df_analyis, x="Salary Maximum", title="Data Analyst Jobs - Minimum Salary", marginal="box", hover_data=df_analyis[["Job Title", "Company Name"]])
fig.show()


# In[55]:


fig = px.histogram(data_frame=df_analyis, x="Salary Average", marginal="box", hover_data=df_analyis[["Job Title", "Company Name"]], title="Average Salary of Data Analyst Jobs")
fig.show()


# In[56]:


data_analyst_title = df_analyis[df_analyis["Job Title"] == "Data Analyst"]
data_analyst_title.head()


# In[57]:


data_analyst_title.describe()


# In[58]:


fig = px.histogram(data_frame=data_analyst_title, x="Salary Minimum", title="Minimum Salary of Data Analyst", marginal="box", hover_data=data_analyst_title[['Job Title', 'python', 'SQL', 'Excel', 'Tableau']])
fig.show()


# In[59]:


fig = px.histogram(data_frame=data_analyst_title, x="Salary Maximum", title="Maximum Salary of Data Analyst", marginal="box", hover_data=data_analyst_title[['Job Title', 'python', 'SQL', 'Excel', 'Tableau']])
fig.show()


# In[74]:


fig = px.histogram(data_frame=data_analyst_title, x="Salary Average", title="Average Salary of Data Analyst", marginal="box", hover_data=data_analyst_title[['Job Title', 'python', 'SQL', 'Excel', 'Tableau']])
fig.show()


# In[61]:


a = df_analyis["Job Title"].value_counts()[:10]
a


# In[62]:


plt.figure(figsize=(6, 4), dpi=(120))
sns.scatterplot(x=a.index, y=a.values)
plt.title("Number of Job Openings by Job Titles")
plt.xlabel("Job Title")
plt.ylabel("Number of Job Openings")
for i, ii in enumerate(a):
    plt.text(i, ii, str(ii))
plt.xticks(rotation=270);


# In[63]:


b = df_analyis["Industry"].value_counts()[:10]
b


# In[64]:


plt.figure(figsize=(8, 4), dpi=(120))
sns.scatterplot(x=b.index, y=b.values)
plt.title("Number of Job Openings by Industry")
plt.xlabel("Industry")
plt.ylabel("Number of Job Openings")
for i, ii in enumerate(b):
    plt.text(i, ii, str(ii), va="center")
plt.xticks(rotation=270);


# In[65]:


c = df_analyis["Sector"].value_counts()[:10]
c


# In[66]:


plt.figure(figsize=(8, 4), dpi=(120))
sns.scatterplot(x=c.index, y=c.values)
plt.title("Number of Job Openings by Sector")
plt.xlabel("Sector")
plt.ylabel("Number of Job Openings")
for i, ii in enumerate(c):
    plt.text(i, ii, str(ii), va="center")
plt.xticks(rotation=270);


# In[67]:


lang_skills = df_analyis[["Job Title", "python", "Excel", "SQL", "Tableau"]]
lang_skills_1 = lang_skills.groupby("Job Title")[["python", "Excel", "SQL", "Tableau"]].sum().sort_values(by="python", ascending=False)[:10]
lang_skills_1['number_of_job_openings'] = df_analyis['Job Title'].value_counts()[:10].values
lang_skills_1


# In[68]:


lang_skills_1.index


# In[69]:


fig = px.bar(data_frame=lang_skills_1, x=lang_skills_1.index, y=["python", "Excel", "SQL", "Tableau"], title="Programming Languages")
fig.show()


# In[70]:


fig = px.scatter(df_analyis, x="Salary Minimum", y="Company Name", color="Rating", hover_data=['Industry', 'Job Title'], 
title = "Minimum Salary by Company Name with Rating Scores")
fig.show()


# In[71]:


fig = px.scatter(df_analyis, x="Salary Average", y="Company Name", color="Rating", hover_data=['Industry', 'Job Title'], 
title = "Average Salary by Company Name with Rating Scores")
fig.show()


# In[ ]:





# In[ ]:




