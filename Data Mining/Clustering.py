from Dataset import *
from Models import *


data, training, testing = loader('3425_data.csv')
training_x = training.drop(columns=['opinionated'])
training_y = training['opinionated']
testing_x = testing.drop(columns=['opinionated'])
testing_y = testing['opinionated']

# -------------------------------------------------------------------
# Clustering
# clustering_data = data[['Q8a', 'Q8d', 'p_age_group_sdc', 'agree_count', 'disagree_count']]
# size_n = len(clustering_data)
# max_k = int(np.sqrt(size_n / 2))
# cluster1 = Clustering(clustering_data, 2)
# cluster2 = Clustering(clustering_data, 5)
# cluster3 = Clustering(clustering_data, max_k)
# -------------------------------------------------------------------
