
#from utils import Utils
#from models import Models 
from random import random
import pandas as pd

from sklearn.cluster import MiniBatchKMeans


if __name__ == "__main__":

    Data_candy_raw = pd.read_csv('../data/raw/candy.csv')

    Data_candy = pd.read_csv('../data/processed/Candy_out.csv')

    kmeansmin = MiniBatchKMeans(n_clusters = 4, batch_size = 8)
    
    kmeansmin.fit(Data_candy)

    #print('total de centros : ', len(kmeansmin.cluster_centers_))
    
    kmeansmin.predict(Data_candy)

    print(kmeansmin.predict(Data_candy))

    Data_candy_raw['group'] = kmeansmin.predict(Data_candy)

    print(Data_candy_raw['group'].unique())
    
    Data_candy_raw.to_csv('../data/processed/candy_vis.csv')
