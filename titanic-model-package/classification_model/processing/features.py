# classification_model/processing/features.py

from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
	"""Extract first letter of variable"""

	def __init__(self, variables):
		if not isinstance(variables, list):
			raise ValueError('variables should be a list')

		self.variables = variables

	def fit(self, X, y=None):
		# we need this step tp fit the sklearn pipeline
		return self

	def transform(self, X):
		# so that we do not overwrite the original dataframe
		X = X.copy()

		for feature in self.variables:
			X[feature] = X[feature].str[0]
			
		return X