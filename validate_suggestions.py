import sys, os, math, re, unicodedata
from timeit import default_timer as timer
from nltk.tag import pos_tag 
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from Classifier import Classifier

start = timer()

#general variables
#MongoDB
host = '192.168.33.10'
port = 27017
username = ''
password = ''
database = 'recsysdb'

APP_NAME = 'Recomender System'
#connecting to MongoDB
def createMongoDBConnection(host, port, username, password, db):
	""" create connection with MongoDB
    Args:
        params to connection
    Returns:
        a connection to db
    """
	client = MongoClient(host, port)
	return client[db]

def calculateRMSE(iduser, ratesRDD):
    db = createMongoDBConnection(host, port, username, password, database)
    products = ratesRDD.map(lambda x: x[0]).collect()
    sumErrors = 0
    for suggestion in db.suggestions.find({'iduser': iduser, 'suggestions.product':{"$in": products}}):
        for product in suggestion['suggestions']:
            rateUser = ratesRDD.filter(lambda x: x[0]==product['product']).map(lambda x: x[1]).collect()
            if len(rateUser) > 0:           
                sumErrors = sumErrors + math.pow(float(product['rate'])-float(rateUser[0]), 2)

    
    return math.sqrt(sumErrors/ratesRDD.count())

def main(sc):
    iduser = 1
    rates =  [
                ('1218194156327', 4),
                ('1196470814718', 4),
                ('1165609791259', 4)
             ]

    RMSE = calculateRMSE(iduser, sc.parallelize(rates))

    print "the RMSE is %f" % RMSE

    elap = timer()-start
    print 'it tooks %d seconds' % elap

if __name__ == '__main__':
    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    main(sc)
