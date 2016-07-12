import sys, os, shutil
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector

class Classifier:

	def __init__(self, sc, classifierType = 'NaiveBayes'):
		self.type = classifierType
		self.sc = sc

	def createVectSpaceCategory(self, featureCorpus, category, tokens):
		self.category = category
		self.tokens = tokens
		numTokens = len(tokens)
		return featureCorpus.map(lambda t: LabeledPoint(category.index(t[2]) , SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])])))

	def createVectSpaceSubcategory(self, featureCorpus, category, tokens):
		self.category = category
		self.tokens = tokens
		numTokens = len(tokens)
		return featureCorpus.map(lambda t: LabeledPoint(category.index((t[2], t[3])) , SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])])))		

	def createVectSpacePost(self, featureCorpus, tokens):
		numTokens = len(tokens)
		return featureCorpus.map(lambda t: (t[0], SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])])))

	def trainModel(self, vectSpace, path):
		try:

			if self.type == 'NaiveBayes':
				model = NaiveBayes.train(vectSpace)
			elif self.type == 'DecisionTree':
				model = DecisionTree.trainClassifier(vectSpace, numClasses = len(self.category), categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=5)

			if not os.path.exists(path):
				os.makedirs(path)
			else:
				shutil.rmtree(path)
				os.makedirs(path)

			model.save(self.sc, path)

		except:
			print "Unexpected error:", sys.exc_info()[0]
		 	raise
		return model

	def getModel(self, path):
		if self.type == 'NaiveBayes':
			return NaiveBayesModel.load(self.sc, path)
		elif self.type == 'DecisionTree':
			return DecisionTreeModel.load(self.sc, path)