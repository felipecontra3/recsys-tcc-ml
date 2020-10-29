import sys, os, math, datetime
from timeit import default_timer as timer
import re, unicodedata
from nltk.tag import pos_tag 
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from Classifier import Classifier
#from base import *

APP_NAME = 'Recomender System - Spark Job'

#general variables
#MongoDB
host = 'localhost'
port = 27017
username = ''
password = ''
database = 'recsysdb_homologacao'

def removeAccents(s):
  s = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
  return re.sub(r'[^\w]', ' ', s)
    
def createMongoDBConnection(host, port, username, password, db):
    client = MongoClient(host, port)
    return client[db]

def findUserById(userId):
    db = createMongoDBConnection(host, port, username, password, database)
    return db.usuario.find_one({'_id': ObjectId(userId)})

def findProductById(prodId):
    db = createMongoDBConnection(host, port, username, password, database)
    return db.produto.find_one({'_id': ObjectId(prodId)})

def findProductsByCategory(categories):
    db = createMongoDBConnection(host, port, username, password, database)
    produtos = db.produto
    product_list = []
    query_filter = {}
    if categories:
      query_filter = {"categorias" : {"$in" : categories}}
    
    print '#### Find products by query {}'.format(query_filter)
    for produto in produtos.find(query_filter):
      keys = produto.keys()
      description = ''
      if 'descricaoLonga' in keys:
          description = removeAccents(description + produto['descricaoLonga'])
      if 'nome' in keys:
          description = removeAccents(description + produto ['nome'])
      id = None
      if '_id' in keys:
          id = str(produto['_id'])
      
      category = ''
      subcategory = ''
      if 'categorias' in keys:
          category = removeAccents(produto['categorias'][0])
          if(len(produto['categorias']) > 1):
              subcategory = removeAccents(produto['categorias'][1])
          
      product_list.append((id, description, category, subcategory))
    
    return product_list

def insertTokensAndCategories(tokens, category, categoryAndSubcategory):
    db = createMongoDBConnection(host, port, username, password, database)

    modelCollection = db.model
    modelCollection.remove({'_type':'token'})

    document_mongo =  dict()
    document_mongo['_type'] = 'token'
    document_mongo['_datetime'] = datetime.datetime.utcnow()
    i = 0
    for t in tokens:
        document_mongo[t] = i
        i = i + 1   

    modelCollection.insert_one(document_mongo)

    modelCollection.remove({'_type':'category'})

    document_mongo =  dict()
    document_mongo['_type'] = 'category'
    document_mongo['_datetime'] = datetime.datetime.utcnow()
    i = 0
    for c in category:
        document_mongo[c] = i
        i = i + 1 

    modelCollection.insert_one(document_mongo)

    modelCollection.remove({'_type':'category and subcategory'})
    
    document_mongo =  dict()
    document_mongo['_type'] = 'category and subcategory'
    document_mongo['_datetime'] = datetime.datetime.utcnow()
    i = 0
    for c in categoryAndSubcategory:
        document_mongo[c[0]+","+c[1]] = i
        i = i + 1 

    modelCollection.insert_one(document_mongo)
    

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
    
def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    token_dict = dict()   
    for token in tokens:
        if token in token_dict:
            token_dict[token] = token_dict[token] + 1
        else:
            token_dict[token] = 1
            
    for t in token_dict:
        token_dict[t] = float(token_dict[t])/float(len(tokens))
        
    return token_dict

def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """
    N = corpus.count()
    uniqueTokens = corpus.flatMap(lambda doc: set(doc[1]))
    tokenCountPairTuple = uniqueTokens.map(lambda t: (t, 1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a,b: a+b)

    return tokenSumPairTuple.map(lambda (k, v): (k, math.log(1.0*N/v)))    


def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens)
    tfIdfDict = {k: v*idfs[k] for k, v in tfs.items()}
    return tfIdfDict

def dotprod(a, b):
    """ Compute dot product
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    dp=0
    for k in a:
        if k in b:
            dp += a[k] * b[k]
    return  dp


def norm(a):
    """ Compute square root of the dot product
    Args:
        a (dictionary): a dictionary of record to value
    Returns:
        norm: a dictionary of tokens to its TF values
    """
    return math.sqrt(dotprod(a, a))
def main(sc):
    start = timer()

    #### 1) Recuperando os produtos da base de dados
    categs = ["Computers & Tablets", "Video Games", "TV & Home Theater"]# , ]
    stpwrds = stopwords.words('portuguese')
    products = findProductsByCategory(categs)
    
    print '####### Creating product rdd with {} product'.format(len(products))
    
    productRDD = sc.parallelize(products)
    #productRDD, discardedProductRDD = entiryProductRDD.randomSplit([2, 8], seed=0L)
   
    
    #### 2) Criadno o corpus de documento utilizando 
    corpusRDD = productRDD.map(lambda s: (s[0], word_tokenize(s[1].lower()), s[2], s[3])).map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], s[2], s[3] )).map(lambda s: (s[0], [x[0] for x in pos_tag(s[1]) if x[1] == 'NN' or x[1] == 'NNP'], s[2], s[3])).cache()

    idfsRDD = idfs(corpusRDD)
    idfsRDDBroadcast = sc.broadcast(idfsRDD.collectAsMap())
    tfidfRDD = corpusRDD.map(lambda x: (x[0], tfidf(x[1], idfsRDDBroadcast.value), x[2], x[3]))
    category = productRDD.map(lambda x: x[2]).distinct().collect()
    categoryAndSubcategory = productRDD.map(lambda x: (x[2], x[3])).distinct().collect()
    tokens = corpusRDD.flatMap(lambda x: x[1]).distinct().collect()

    insertTokensAndCategories(tokens, category, categoryAndSubcategory)
    
    classifier = Classifier(sc, 'NaiveBayes')   
    
    
    trainingVectSpaceCategoryRDD, testVectSpaceCategoryRDD = classifier.createVectSpaceCategory(tfidfRDD, category, tokens).randomSplit([8, 2], seed=0L)
    modelNaiveBayesCategory = classifier.trainModel(trainingVectSpaceCategoryRDD, '/dados/models/naivebayes/category_new')
    predictionAndLabelCategoryRDD = testVectSpaceCategoryRDD.map(lambda p : (category[int(modelNaiveBayesCategory.predict(p.features))], category[int(p.label)]))
    acuraccyCategory = float(predictionAndLabelCategoryRDD.filter(lambda (x, v): x[0] == v[0]).count())/float(predictionAndLabelCategoryRDD.count())
    print 'the accuracy of the Category Naive Bayes model is %f' % acuraccyCategory

    #trainingVectSpaceSubcategory, testVectSpaceSubcategory = classifier.createVectSpaceSubcategory(tfidfRDD, categoryAndSubcategory, tokens).randomSplit([8, 2], seed=0L)
    #modelNaiveBayesSubcategory = classifier.trainModel(trainingVectSpaceSubcategory, '/dados/models/naivebayes/subcategory_new')

    #predictionAndLabelSubcategory = testVectSpaceSubcategory.map(lambda p : (categoryAndSubcategory[int(modelNaiveBayesSubcategory.predict(p.features))], categoryAndSubcategory[int(p.label)]))
    #acuraccySubcategory = float(predictionAndLabelSubcategory.filter(lambda (x, v): x[0] == v[0]).count())/float(predictionAndLabelSubcategory.count())
    print 'the accuracy of the Subcategory Naive Bayes model is %f' % acuraccySubcategory

    #test with DecisionTree Model
    #classifierDT = Classifier(sc, 'DecisionTree')
    #trainingVectSpaceCategory, testVectSpaceCategory = classifierDT.createVectSpaceCategory(tfidfRDD, category, tokens).randomSplit([8, 2], seed=0L)
    #modelDecisionTreeCategory = classifierDT.trainModel(trainingVectSpaceCategory, '/dados/models/dt/category_new')

    #predictions = modelDecisionTreeCategory.predict(testVectSpaceCategory.map(lambda x: x.features))
    #predictionAndLabelCategory = testVectSpaceCategory.map(lambda lp: lp.label).zip(predictions)
    #acuraccyDecisionTree = float(predictionAndLabelCategory.filter(lambda (x, v): x == v).count())/float(predictionAndLabelCategory.count())   
    #print 'the accuracy of the Decision Tree model is %f' % acuraccyDecisionTree

    elap = timer()-start
    print 'it tooks %d seconds' % elap

if __name__ == '__main__':
    conf = (SparkConf().setAppName("RECSYS"))#.set("spark.driver.memory", "14G").set("spark.executor.memory", "14G").set("spark.python.worker.memory", "14G").set("spark.cores.max",8).set("spark.driver.extraJavaOptions", "-Xmx14G -Xms14G"))
    sc = SparkContext(conf=conf)
    main(sc)
