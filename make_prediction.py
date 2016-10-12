import sys, os, math, re, unicodedata, click
from timeit import default_timer as timer
from nltk.tag import pos_tag 
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from bson.objectid import ObjectId
from pyspark import SparkConf, SparkContext
from langdetect import detect
#from base import *
# encoding=utf8  

from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.sql import Row, SQLContext
from pyspark.sql.functions import col

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

def removeAccents(s):
  s = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
  return re.sub(r'[^\w]', ' ', s)
    
def createMongoDBConnection(host, port, username, password, db):
	client = MongoClient(host, port)
	return client[db]

def findUserById(userId):
    db = createMongoDBConnection(host, port, username, password, database)
    return db.users.find_one({'_id': ObjectId(userId)})

def findPosts(user):
    posts = []

    if user['facebook']['posts'] is not None:
        for post in user['facebook']['posts']:
            if 'message' in post.keys():
                posts.append((post['id_post'], removeAccents(post['message']), u'Post', u'Facebook'))

    if user['twitter'] is not None:
        for post in user['twitter']:
            if 'text' in post.keys():
                posts.append((post['id'], removeAccents(post['text']), u'Post', u'Twitter'))  

    return posts


def findProductById(prodId):
    db = createMongoDBConnection(host, port, username, password, database)
    prod = db.produto.find_one({'_id': ObjectId(prodId)})
    prod['idprod'] = str(prod['_id'])
    return prod

def updateUser(user):
    db = createMongoDBConnection(host, port, username, password, database)
    return db.usuario.save(user)

def getTokensAndCategories():  
    db = createMongoDBConnection(host, port, username, password, database)
    model = db.model
    
    tokens_dict = db.model.find({"_type": "token"}).limit(1).next()
    del tokens_dict['_type']
    del tokens_dict['_id']
    del tokens_dict['_datetime']
    tokens_list = [None] * (max(tokens_dict.values()) + 1)

    for key, value in tokens_dict.iteritems():
        tokens_list[value] = key

    categories_dict = db.model.find({"_type": "category"}).limit(1).next()
    del categories_dict['_type']
    del categories_dict['_id']
    del categories_dict['_datetime']
    categories_list = [None] * (max(categories_dict.values()) + 1)

    for key, value in categories_dict.iteritems():
        categories_list[value] = key

    categories_and_subcategories_dict = db.model.find({"_type": "category and subcategory"}).limit(1).next()
    del categories_and_subcategories_dict['_type']
    del categories_and_subcategories_dict['_id']
    del categories_and_subcategories_dict['_datetime']
    categories_and_subcategories_list = [None] * (max(categories_and_subcategories_dict.values()) + 1)

    for key, value in categories_and_subcategories_dict.iteritems():
        pre_string = key.split(",")
        categories_and_subcategories_list[value] = (pre_string[0], pre_string[1])

    return tokens_list, categories_list, categories_and_subcategories_list
    
def insertSuggestions(suggestions, iduser, posts):   
    recomendations = dict()
    recomendations['recomendacoes'] = []    

    for post in suggestions:

        suggestions_dict = dict()

        suggestions_dict['postId'] = post[0][0]
        suggestions_dict['products'] = []

        for post_base in posts:
            #isso nao esta funcionando, verificar o pq
            if int(post_base[0]) == int(post[0][0]):
                suggestions_dict['post'] = post_base

        for product in post:
            if len(product) > 0:
                prod = findProductById(product[1])
                if len(prod) > 0:
                    prod['cosineSimilarity'] = product[2]
                    suggestions_dict['products'].append(prod)            

        recomendations['recomendacoes'].append(suggestions_dict)            

    db = createMongoDBConnection(host, port, username, password, database)
    db.users.update_one({"_id": ObjectId(iduser)}, {"$set" : recomendations})

    return True

def cossine(v1, v2):
    if (v1.dot(v1)*v2.dot(v2)) != 0:
        return v1.dot(v2)/(v1.dot(v1)*v2.dot(v2))
    else:
        return 0



def main(sc, sqlContext):

    #print '---Pegando usuario, posts, tokens e categorias do MongoDB---'
    start_i = timer()
    user = findUserById(iduser)
    posts = findPosts(user) 
    
    tokens, category, categoryAndSubcategory = getTokensAndCategories()
    postsRDD = (sc.parallelize(posts).map(lambda s: (s[0], word_tokenize(s[1].lower()), s[2], s[3]))
                    .map(lambda p: (p[0], [x for x in p[1] if x in tokens] ,p[2], p[3]))
                    .cache())

    #print '####levou %d segundos' % (timer() - start_i)

    #print '---Pegando produtos do MongoDB---'
    start_i = timer()

    #print '####levou %d segundos' % (timer() - start_i)
    
    #print '---Criando corpusRDD---'
    start_i = timer()
    stpwrds = stopwords.words('portuguese')
    corpusRDD = (postsRDD.map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], s[2], s[3]))
                         .filter(lambda x: len(x[1]) >= 20 or (x[2] == u'Post' and len(x[1])>0))
                         .cache())

    #print '####levou %d segundos' % (timer() - start_i)

    #print '---Calculando TF-IDF---'
    start_i = timer()
    wordsData = corpusRDD.map(lambda s: Row(label=int(s[0]), words=s[1], type=s[2]))
    
    wordsDataDF = sqlContext.createDataFrame(wordsData).unionAll(sqlContext.read.parquet("/home/felipe/Documentos/TCC/Experimento/spark_cluster/spark-1.6.2-bin-hadoop2.6/wordsDataDF.parquet"))

    numTokens = len(tokens)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=numTokens)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    featurizedData = hashingTF.transform(wordsDataDF)

    idfModel = idf.fit(featurizedData)
    tfIDF = idfModel.transform(featurizedData).cache()

    postTFIDF = tfIDF.filter(tfIDF.type==u'Post').cache()

    #print '####levou %d segundos' % (timer() - start_i)

    #print '---Carregando modelo---'
    start_i = timer()
    model = NaiveBayesModel.load(sc, '/dados/models/naivebayes/modelo_categoria')
    #print '####levou %d segundos' % (timer() - start_i)

    #print '---Usando o modelo---'
    start_i = timer()
    predictions = postTFIDF.map(lambda p: ( model.predict(p.features), p[0])).groupByKey().mapValues(list).collect()    
    #print '####levou %d segundos' % (timer() - start_i)

    #print '---Calculando similaridades---'
    start_i = timer()
    suggestions = []

    for prediction in predictions:
        category_to_use = category[int(prediction[0])]
        #print ' Calculando similaridades para a categoria: {}'.format(category_to_use)
        tf = tfIDF.filter(tfIDF.type==category_to_use).cache()
        for post in prediction[1]:
            postVector = postTFIDF.filter(postTFIDF.label == post).map(lambda x: x.features).collect()[0]
            sim = (tf
                    .map(lambda x: (post, x.label, cossine(x.features, postVector)))
                    .filter(lambda x: x[2]>=threshold)
                    .collect())
            if len(sim) > 0:
                suggestions.append(sim)

    #print '####levou %d segundos' % (timer() - start_i)

    if len(suggestions) > 0:
        #print '---Inserindo recomendacoes no MongoDB---'
        start_i = timer()
        insertSuggestions(suggestions, iduser, posts)
        #print '####levou %d segundos' % (timer() - start_i)

if __name__ == '__main__':

    APP_NAME = 'Recomender System - Calculo de recomendacao'
    threshold  = 0.002
    #numMaxSuggestionsPerPost = 5
    numStarts = 5

    host = 'localhost'
    port = 27017
    username = ''
    password = ''
    database = 'tcc-recsys-mongo'

    start = timer()
    iduser = sys.argv[1]

    sc = SparkContext(appName=APP_NAME)
    sqlContext = SQLContext(sc)

    main(sc, sqlContext)