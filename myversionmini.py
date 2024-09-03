#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random


# In[3]:


get_ipython().run_line_magic('cd', 'C:\\Users\\Pranav Gopal N V\\Downloads\\archive (9)')


# In[4]:


t=pd.read_csv("tv_shows.csv")


# In[5]:


t.head()


# In[6]:


t.dtypes


# In[7]:


import re


# In[8]:


t['IMDb'].isnull().sum()


# In[9]:


t['IMDb']= t['IMDb'].replace(np.nan, 0)


# In[10]:


t.head()


# In[11]:


t["Age"]=t["Age"].str.replace("+","",regex=False)


# In[12]:


t['Age']=pd.to_numeric(t['Age'],errors='coerce')


# In[13]:


t["Rotten Tomatoes"]=t["Rotten Tomatoes"].str.replace("/",".0")
t['Rotten Tomatoes']=pd.to_numeric(t['Rotten Tomatoes'],errors='coerce')


# In[14]:


t["IMDb"]=t["IMDb"].str.replace("/","") 
t['IMDb']=pd.to_numeric(t['IMDb'],errors='coerce')


# In[15]:


t.head()


# In[16]:


plt.subplots(figsize=(4,6))
sns.histplot(t["Year"],kde=False, color="blue")


# In[17]:


print("TV Shows with highest IMDb ratings are= ")
print((t.sort_values("IMDb",ascending=False).head(20))['Title'])


# In[18]:


print("TV Shows with highest Rotten Tomatoes scores are= ")
print((t.sort_values("Rotten Tomatoes",ascending=False).head(20))['Title'])


# In[19]:


ratings=t[["Title",'IMDb',"Rotten Tomatoes"]]
ratings.head()


# In[20]:


len(ratings)


# In[21]:


ratings.info()


# In[22]:


ratings=ratings.dropna()


# In[23]:


ratings["IMDb"]=ratings["IMDb"]*10


# In[24]:


ratings.head()


# In[25]:


X=ratings[["IMDb","Rotten Tomatoes"]]


# In[26]:


X.head()


# In[27]:


plt.figure(figsize=(10,6))
sns.scatterplot(x = 'IMDb',y = 'Rotten Tomatoes',  data = X  ,s = 70 )
plt.xlabel('IMDb rating (multiplied by 10)')
plt.ylabel('Rotten Tomatoes') 
plt.title('IMDb rating (multiplied by 10) vs Rotten Tomatoes Score')
plt.show()


# In[28]:


from sklearn.cluster import KMeans


# In[29]:


wcss=[]

for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)


# In[32]:


plt.figure(figsize=(12,6))

plt.plot(range(1,11),wcss)

plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")

plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")

plt.show()


# In[33]:


km=KMeans(n_clusters=4)


# In[34]:


km.fit(X)


# In[35]:


y=km.predict(X)


# In[36]:


ratings["label"]=y


# In[37]:


ratings.head()


# In[38]:


plt.figure(figsize=(10,6))
sns.scatterplot(x = 'IMDb',y = 'Rotten Tomatoes',hue="label",  
                 palette=['green','orange','red','blue'], legend='full',data = ratings  ,s = 60 )

plt.xlabel('IMDb rating(Multiplied by 10)')
plt.ylabel('Rotten Tomatoes score') 
plt.title('IMDb rating(Multiplied by 10) vs Rotten Tomatoes score')
plt.show()


# In[39]:


print('Number of Cluster 0 TV Shows are=')
print(len(ratings[ratings["label"]==0]))
print("--------------------------------------------")
print('Number of Cluster 1 TV Shows are=')
print(len(ratings[ratings["label"]==1]))
print("--------------------------------------------")
print('Number of Cluster 2 TV Shows are=')
print(len(ratings[ratings["label"]==2]))
print("--------------------------------------------")
print('Number of Cluster 3 TV Shows are=')
print(len(ratings[ratings["label"]==3]))
print("--------------------------------------------")


# In[40]:


print('TV Shows in cluster 0')

print(ratings[ratings["label"]==0]["Title"].values)


# In[41]:


print('TV Shows in cluster 0')

for title in ratings[ratings["label"] == 0]["Title"].values:
    print(title)


# In[42]:


print('TV Shows in cluster 1')

for title in ratings[ratings["label"] == 1]["Title"].values:
    print(title)


# In[43]:


print('TV Shows in cluster 2')

for title in ratings[ratings["label"] == 2]["Title"].values:
    print(title)


# In[44]:


print('TV Shows in cluster 3')

for title in ratings[ratings["label"] == 3]["Title"].values:
    print(title)


# In[ ]:





# In[ ]:




