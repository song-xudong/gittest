# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:42:11 2022

作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com

"""


import pandas as pd

df = pd.read_csv('airport.csv',engine='python')
df.head()

#%%

from matplotlib import pyplot

pyplot.hist(df['overall_rating'].dropna(), bins=10, rwidth=0.9, color ='red')
pyplot.title('Airport Rating Distribution')
pyplot.xlabel('Rating')
pyplot.ylabel('Count')
pyplot.show() 


#%%


from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[2, 0], [1, 3], [4, 3]])
print('Original Matrix A: \n', A)
# SVD
U, Sigma, VT = svd(A)
print('Matrix U: \n', U)
print('Matrix Sigma: \n', Sigma)
print('Matrix VT: \n', VT)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)

#%%


from nltk.stem import PorterStemmer #Stemming Package  
import re  #Regular expression operation package

porter = PorterStemmer()

documents = df['content']
Cleaned_doc = []
for r in range(len(documents)):
    review = documents[r]
    try:
        
        review = re.sub('[^A-Za-z]', ' ', review) 
        
        review = review.lower()
        
        Tokens = review.split()

        Filtered_token = [w for w in Tokens if len(w)>3] 
        review = ' '.join(Filtered_token)        
    except:
        continue
    
    Cleaned_doc.append(review)  
    print('-[Review Text]: ', review)

#%%


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

for r in range(len(Cleaned_doc)):
    each_item = []
    for t in Cleaned_doc[r].split():
        if t not in stop_words:
             each_item.append(t)
    Cleaned_doc[r] = ' '.join(each_item) 
    print('-[Cleaned Text]: ', Cleaned_doc[r])
    
    
#%%

   
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features= 1000, # keep top 1000 terms 
                             max_df = 0.5, 
                             smooth_idf=True)

A = vectorizer.fit_transform(Cleaned_doc)
A.shape 
    
#%%

from sklearn.decomposition import TruncatedSVD

svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(A)
print("Number of Components:", len(svd_model.components_))

#%%

from matplotlib import pyplot
TopicWeight = svd_model.singular_values_

pyplot.figure(figsize=(8, 4)) 
pyplot.bar([x for x in range(len(TopicWeight))],TopicWeight)
pyplot.suptitle('Topic Weight - (Singular values)')
pyplot.xlabel('Topic')
pyplot.ylabel('Weight')
pyplot.xticks([x for x in range(len(TopicWeight))]);

    
#%%

print('Length of each component: ', len(svd_model.components_[0]))
print('Value of  the first component (Weights of words):\n ', svd_model.components_[1]) 


#%%

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, abs(comp))
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:5]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0], ':', '{0:.3f}'.format(t[1]))
    print(" ")
    

#%%

from wordcloud import WordCloud
import math

rows = math.ceil(len(svd_model.components_)/4)
fig, ax = pyplot.subplots(rows, 4, figsize=(15,2.5*rows))
[axi.set_axis_off() for axi in ax.ravel()]

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, abs(comp))
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:5] 
    
    Word_Frequency = dict(sorted_terms[0:10])
    
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(Word_Frequency)
    
    subfig_Row = math.floor(i/4)
    subfig_Col = math.ceil(i%4)
    ax[subfig_Row,subfig_Col].imshow(wordcloud)
    ax[subfig_Row,subfig_Col].set_title("Topic {}".format(i+1)) 
        
pyplot.show() 

#%%

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()# Fit and transform the processed titles

count_data = count_vectorizer.fit_transform(Cleaned_doc)
count_data

#%%


terms = count_vectorizer.get_feature_names()

total_counts = np.zeros(len(terms))
for t in count_data:
    total_counts+=t.toarray()[0]

count_dict = (zip(terms, total_counts))
count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:40]
    
words = [w[0] for w in count_dict]
counts = [w[1] for w in count_dict]
x_pos = np.arange(len(words))
    
plt.figure(2, figsize=(15, 4))
plt.subplot(title='40 most common words')
sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, counts, palette='husl')
plt.xticks(x_pos, words, rotation=90) 
plt.xlabel('words')
plt.ylabel('counts')
plt.show()


#%%

keepIndex = [];
for t in range(len(total_counts)):
    if total_counts[t] < 1000 and total_counts[t] > 50:
        keepIndex.append(t)

print('Number of Terms Remained: ', len(keepIndex))

ReducedTerm = [terms[t] for t in keepIndex]
ReducedCount = count_data[:,keepIndex] 
ReducedCount

#%%


from sklearn.decomposition import LatentDirichletAllocation as LDA

number_topics = 10
 
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(ReducedCount)
#Trained LDA model
lda.components_     
    
    
#%%

Word_Topics_Pro = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
print(Word_Topics_Pro) 

 
#%%
 

for topic_idx, topic in enumerate(Word_Topics_Pro):
    print("\nTopic #%d:" % topic_idx)
    count_dict = (zip(ReducedTerm, topic))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:5]
    for w in count_dict:
        print(w[0], ': {0:.3f}'.format(w[1]))    
 
    
#%%

df_topic_keywords = pd.DataFrame(Word_Topics_Pro)
df_topic_keywords.columns = ReducedTerm
df_topic_keywords  
    
 
#%%

TopicDis_Doc = lda.transform(ReducedCount)
df_document_topics = pd.DataFrame(TopicDis_Doc)
df_document_topics    

#%%

TopicDis_Doc = TopicDis_Doc[0:5]
print('Topic Probablity distribution by Document: \n', TopicDis_Doc)
Bar_index = np.asarray(range(1,number_topics+1))

pyplot.figure(figsize=(9,5)) 
pyplot.title('Topic Probaility Distribution by Document', fontsize=16)
pyplot.xlabel('Topic')
pyplot.ylabel('Probability')

width = 0.15   
for i in range(0,5):
    pyplot.bar(Bar_index + i*width, TopicDis_Doc[i].tolist(), width,  label='doc ' + str(i))
    
pyplot.xticks(Bar_index + 2*width, Bar_index)
pyplot.legend()
pyplot.show();


#%%


import numpy as np

ReducedTerm_Selected = ReducedCount[np.where(df['author_country'] == 'United Kingdom')]
TopicDis_Doc = lda.transform(ReducedTerm_Selected)

Overall_Topic_Dis = sum(TopicDis_Doc)/sum(sum(TopicDis_Doc))
    
    
pyplot.figure(figsize=(7,4)) 
pyplot.title('Topic Distribution of Document Group', fontsize=16)
pyplot.xlabel('Topic')
pyplot.ylabel('Probability')
pyplot.bar(Bar_index, Overall_Topic_Dis.tolist(), 0.3,  label='United Kingdom')
pyplot.xticks(Bar_index, Bar_index)
pyplot.legend()
pyplot.show();


#%%


Cleaned_doc_new = []
print('CLEANED TEXT NEW: ')
for r in range(len(Cleaned_doc)):
    each_item = []
    for t in Cleaned_doc[r].split():
        if t in ReducedTerm:
             each_item.append(t)
    Cleaned_doc_new.append(each_item) 
    print(Cleaned_doc_new[r])

#%%


import gensim.corpora as corpora

id2word = corpora.Dictionary(Cleaned_doc_new)
print(id2word.token2id)

#%%


Corpus = [id2word.doc2bow(text) for text in Cleaned_doc_new]
print(Corpus)  

#%%


import gensim
from gensim.models.ldamodel import LdaModel
from pprint import pprint#

lda_model = gensim.models.ldamodel.LdaModel(corpus=Corpus,
                                       id2word=id2word,
                                       num_topics=10,
                                       random_state=100)

pprint(lda_model.print_topics(num_words=10))
doc_lda = lda_model[Corpus]


#%%


from gensim.models import CoherenceModel


coherence_model_lda = CoherenceModel(model=lda_model, 
                                     texts=Cleaned_doc_new, 
                                     dictionary=id2word, 
                                     coherence='c_v')


coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

#%%

Topics = list(range(1,11,1))
coherence_scores = []
Trained_Models = []
for top in Topics:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=Corpus,
                                               id2word=id2word,
                                               num_topics=top,
                                               random_state=100)
    
    Trained_Models.append(lda_model)
    
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=Cleaned_doc_new, 
                                         dictionary=id2word, 
                                         coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
   
    coherence_scores.append(coherence)
    print('Topic Number: {0} -- Coherence: {1}'.format(top, coherence))
    
    
#%%


pyplot.plot(coherence_scores)
pyplot.xticks(range(0,len(Topics)),Topics)
pyplot.title('Coherence Score by Topic Number', fontsize=16)
pyplot.xlabel('Topic Number')
pyplot.ylabel('Coherence')

#%%

import numpy
lda_model = Trained_Models[numpy.argmax(coherence_scores)]

lda_model.show_topics(num_words=10)









    
