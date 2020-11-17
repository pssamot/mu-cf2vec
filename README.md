# mu-cf2vec
  Code used in μ-cf2vec: Representation Learning in Personalized Algorithm Selection in Recommender Systems.
  
  
  The dataset used was the Movie Lens 20M (https://grouplens.org/datasets/movielens/20m/). Since this is a meta learning task. We divided the data in three partitions and we selected the partitions with the best balance between the number of users belonging to the class zeroes and the value K. 
 
  70_10_20 was the choosen one. We obtained the embeddings for each user (training interactions) using two methods: Variational Auto Encoders for Collaborative Filtering, and Collaborative Denoising Auto Encodes. 
  
 The processed data and the embbedings for each user can be find in: https://drive.google.com/file/d/1yYur8FVU-RBw4_5wpl9lLYX6UfKeD64G/view?usp=sharing
 
 We used the Fast Python Collaborative Filtering for Implicit Datasets (https://github.com/benfred/implicit). Base Learners used:
  - ALS
  - BPR
  - LMF
  - KNN
  - Most popular (baseline)

 For the metalearners we used:
  - Logistic Regression
  - Multi Layer Perceptron
  - Support Vector Machine
  - Random Forest
  - Light GBM
 
Evaluation Method: 5-fold cross validation
