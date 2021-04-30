import pandas as pd
import glob, os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

path = r'C:\\Users\\HANA\\PycharmProjects\\HANATOUR\\NLP\\TEXT_Example\\topic'
all_files = glob.glob(os.path.join(path,"*.data"))
filename_list = []
opinion_text = []

for file_ in all_files:
    df = pd.read_table(file_, index_col=None, header=0, encoding='latin1')
    filename_ = file_.split('\\')[-1]
    filename = filename_.split('.')[0]

    filename_list.append(filename)
    opinion_text.append(df.to_string())

document_df = pd.DataFrame({'filename':filename_list, 'opinion_text':opinion_text})
# print(document_df.head())

from sklearn.feature_extraction.text import TfidfVectorizer
import Common_Module.CMNLP as CMNLP

tfodf_vect = TfidfVectorizer(tokenizer=CMNLP.LemNormalize, stop_words='english', ngram_range=(1,2), min_df=0.05, max_df=0.85)
feature_vect = tfodf_vect.fit_transform(document_df['opinion_text'])
# print(feature_vect)

from sklearn.cluster import KMeans

km_cluster = KMeans(n_clusters=5, max_iter=1000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_

document_df['cluster_label'] = cluster_label

# print(document_df.head())
# print(document_df[document_df['cluster_label']==0].sort_values(by='filename'))
# print(document_df[document_df['cluster_label']==1].sort_values(by='filename'))
# print(document_df[document_df['cluster_label']==2].sort_values(by='filename'))
# print(document_df[document_df['cluster_label']==3].sort_values(by='filename'))
# print(document_df[document_df['cluster_label']==4].sort_values(by='filename'))

cluster_centers = km_cluster.cluster_centers_
print('cluster centers shape :', cluster_centers.shape)
print(cluster_centers)

feature_names = tfodf_vect.get_feature_names()
cluster_details = CMNLP.get_cluster_details(cluster_model=km_cluster, cluster_data=document_df, feature_names=feature_names, clusters_num=5, top_n_features=10)
CMNLP.print_cluster_details(cluster_details)