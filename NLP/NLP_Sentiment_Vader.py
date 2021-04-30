from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import pandas as pd
import re
import nltk
import numpy as np

# 첫 1회만 실행
# nltk.download('all')

review_df = pd.read_csv('C:\\Users\\HANA\\PycharmProjects\\HANATOUR\\NLP\\TEXT_Example\\labeledTrainData.tsv', header=0, sep="\t", quoting=3)
# print(review_df.head())
# review_df['review'] = review_df['review'].str.replace('<br />', ' ')
# review_df['review'] = review_df['review'].apply(lambda x : re.sub("[^a-zA-Z]"," ", x))
print(review_df['review'][0])
print(review_df['sentiment'][0])
print(review_df['id'][0])

senti_analyzer = SentimentIntensityAnalyzer()
senti_scores = senti_analyzer.polarity_scores(review_df['review'][0])
print(senti_scores)

def vader_polarity(review, threshold=0.1):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)

    agg_score = scores['compound']
    final_sendtiment = 1 if agg_score >= threshold else 0
    return final_sendtiment

review_df['vader_preds'] = review_df['review'].apply(lambda x : vader_polarity(x,0.1))
y_target = review_df['sentiment'].values
vader_preds = review_df['vader_preds'].values

print(y_target)
# print(np.round(acc))