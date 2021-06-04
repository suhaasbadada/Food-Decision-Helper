from flask import Flask,render_template,request,redirect,jsonify
import pandas as pd
import numpy as np
#from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop = set(stopwords.words('english'))
choices=pd.read_csv('mood_choices.csv')

#data_url='https://www.dropbox.com/s/jr0j280reh1ydgx/data_cleaned.csv?dl=1'
#data=pd.read_csv(data_url)
#df_percent=data.head(int(0.1*data.shape[0]+3))

df_percent=pd.read_csv('df_percent.csv')
df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)
urldf=pd.read_csv('urldf.csv')
df_percent['url']=list(urldf['url'])

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'].apply(lambda x: np.str_(x)))

from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

app=Flask(__name__)

@app.route('/') 

def Home():
    return render_template('index.html')

@app.route('/omood/<mood>', methods=['POST'])
def recommend_me(mood):
    lemmatizer = WordNetLemmatizer()
    foodcount = {}
    for i in range(124):
        temp = [temps.strip().replace('.','').replace(',','').lower() for temps in str(choices["comfort_food_reasons"][i]).split(' ') if temps.strip() not in stop ]
        if mood in temp:
            foodtemp = [lemmatizer.lemmatize(temps.strip().replace('.','').replace(',','').lower()) for temps in str(choices["comfort_food"][i]).split(',') if temps.strip() not in stop ]
            for a in foodtemp:
                if a not in foodcount.keys():
                    foodcount[a] = 1 
                else:
                    foodcount[a] += 1
    sorted_food = []
    sorted_food = sorted(foodcount, key=foodcount.get, reverse=True)

    return sorted_food


@app.route('/omood',methods=['POST'])
def mood_input():
    userMood=request.form["userMood"]
    fr=recommend_me(userMood)
    if(userMood):
        return render_template('omood.html',fr=fr)
    else:
        return render_template('imood.html')


@app.route('/imood',methods=['GET', 'POST']) 
def Mood():
    return render_template('imood.html')

@app.route('/ivisited',methods=['GET', 'POST']) 
def Visited():
    return render_template('ivisited.html')

@app.route('/ovisited/<name>/<cosine_similarities>', methods=['POST'])
def recommend(name, cosine_similarities = cosine_similarities):

    recommend_restaurant = []

    idx = indices[indices == name].index[0]

    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    top30_indexes = list(score_series.iloc[0:31].index)

    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])

    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost','url'])
    
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost','url']][df_percent.index == each].sample()))
    
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    #print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new

@app.route('/ovisited',methods=['POST'])
def visited_input():
    userVisited=request.form["userVisited"]
    if(userVisited):
        rr=recommend(userVisited)
        return render_template('ovisited.html',rr_html=rr.to_html(escape=False))
    else:
        return render_template('/ivisited.html')

