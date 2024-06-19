#!/usr/bin/env python
# coding: utf-8

# In[85]:


get_ipython().system('pip install lenskit')


# In[86]:


from lenskit.algorithms import Recommender, user_knn as knn
from lenskit import batch, topn, util
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import gzip
import json


# In[87]:


# Function to down-sample the dataset
def downsample_data(ratings, percentage):
    sampled_ratings = ratings.sample(frac=percentage, random_state=1)
    return sampled_ratings


# In[88]:


# Function to load the JSON data
def load_json_data(file_path):
    with gzip.open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


# In[89]:


file_path = 'E:/Mechatronics_Uni Siegen Courses/Master Thesis/datasets/Amazon/Toys_and_Games_5.json.gz'
ratings = load_json_data(file_path)


# In[90]:


ratings = ratings.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating'})
ratings = ratings.dropna(subset=['rating'])


# In[91]:


# Keep only the necessary columns
ratings = ratings[['user', 'item', 'rating']]


# In[92]:


# Handle duplicate ratings by averaging them
ratings = ratings.groupby(['user', 'item']).agg({'rating': 'mean'}).reset_index()


# In[93]:


# Down-sample the dataset to different percentages (adjust percentage as needed)
ratings = downsample_data(ratings, percentage=0.9)


# In[94]:


# Split data into training, validation, and test sets (60%, 20%, 20%)
train_data, remaining_data = train_test_split(ratings, test_size=0.4, random_state=1)
valid_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=1)


# In[95]:


# Initialize UserUser recommender algorithm
algo_uu = knn.UserUser(10)


# In[96]:


# Training the model
fittable = util.clone(algo_uu)
fittable = Recommender.adapt(fittable)
fittable.fit(train_data)


# In[97]:


# Save the model with a unique name
joblib.dump(fittable, 'useruser_model.pkl')


# In[98]:


# Function to evaluate the algorithm on validation set
def evaluate_algorithm(algo, valid, Rec_Num):
    users = valid.user.unique()
    recs = batch.recommend(algo, users, Rec_Num)
    recs['Algorithm'] = 'UserUser'

    # Compute nDCG on validation set
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, valid)
    nDCG_mean = results.groupby('Algorithm').ndcg.mean().iloc[0]

    return nDCG_mean


# In[99]:


Rec_Num = 10
nDCG_validation = evaluate_algorithm(fittable, valid_data, Rec_Num)
print("Validation nDCG @", Rec_Num, "for UserUser:" ,nDCG_validation)


# In[100]:


# Function to load the model and evaluate on test set
def evaluate_on_test_set(test, Rec_Num):
    # Load the model
    loaded_model = joblib.load('useruser_model.pkl')

    # Evaluate on test set
    test_users = test.user.unique()
    test_recs = batch.recommend(loaded_model, test_users, Rec_Num)
    test_recs['Algorithm'] = 'UserUser'

    # Compute nDCG on test set
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    test_results = rla.compute(test_recs, test)
    test_nDCG_mean = test_results.groupby('Algorithm').ndcg.mean().iloc[0]

    return test_nDCG_mean


# In[101]:


# Evaluate on test set whenever needed
nDCG_test = evaluate_on_test_set(test_data, Rec_Num)
print(f"Test nDCG for UserUser: {nDCG_test}")


# In[102]:


import os
print(os.getcwd())


# In[ ]:




