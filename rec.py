import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import NearestNeighbors

video = pd.read_csv('USvideos.csv',  error_bad_lines=False,encoding="latin-1")
video.columns = ['video_id', 'trending_date' ,'title', 'channel_title','category_id', 'publish_time','tags' ,'views', 'likes', 'dislikes',   
'comment_count', 'thumbnail_link','comments_disabled','ratings_disabled','video_error_or_removed',' description']
user = pd.read_csv('user.csv',  error_bad_lines=False,encoding="latin-1")
user.columns = ['userID', 'Location' ,'Age']
rating = pd.read_csv('rating.csv',  error_bad_lines=False,encoding="latin-1")
rating.columns = ['userID', 'video_id' ,'Rating']
print(rating.shape)
print rating.head()
 
 
 
 
 
 
 
print(rating.shape)
plt.rc("font", size=15)
rating.Rating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()



combine_video_rating=pd.merge(rating,video, on='video_id')
columns=['channel_title','publish_time' ,'tags', 'thumbnail_link' ,'comments_disabled','ratings_disabled','video_error_or_removed']
combine_video_rating=combine_video_rating.drop(columns, axis=1)
print combine_video_rating.head()
combine_video_rating=combine_video_rating.dropna(axis=0, subset=['title'])

video_rating_count=(combine_video_rating.groupby(by=['title'])['Rating'].count().reset_index().rename(columns={'Rating':'totalRatingCount'})
[['title','totalRatingCount']])









user.Age.hist(bins=[0, 10, 20, 30, 40, 50,60,80])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('system2.png', bbox_inches='tight')
plt.show()








rating_with_totalRatingCount = combine_video_rating.merge(video_rating_count, left_on = 'title', right_on = 'title', how = 'left')
rating_with_totalRatingCount.head()

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(video_rating_count['totalRatingCount'].describe())

print(video_rating_count['totalRatingCount'].quantile(np.arange(.9, 1, .01)))


popularity_threshold = 81
rating_popular_video = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
print rating_popular_video.head()





combined = rating_popular_video.merge(user, left_on = 'userID', right_on = 'userID', how = 'left')

us_canada_user_rating = combined[combined['Location'].str.contains("Bangladesh|USA")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
us_canada_user_rating.head()








us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'title'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'title', columns = 'userID', values = 'Rating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)



model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)







query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 10)

for i in range(0, 4):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {1}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))








