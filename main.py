import copy 
import math


class TFIDFTest:
	def create_words_set(self, corpus: list[str]) -> set[str]:
		"""
		Create a set of the unique words in the corpus.

		Arg:
			corpus (list[str]): the corpus containing all the "documents".
		Returns:
			unique set of words (set[str])
		"""
		words_set = set()
		for doc in  corpus:
			words = doc.split(" ")
			words_set = words_set.union(set(words))
		return words_set

	def create_tf(self, words_set: set[str], corpus: list[str]) -> dict:
		"""
		Create the term frequency 

		Args: 
			words_set (set[str]): the set of unique words
			corpus (list[str]): the corpus containing all the "documents".
		Returns: 
			tf (dict): dict with doc # as key and word frequency as value. I.E {0:{"word1": 0, "word2": 0.125}, 1:...}
		"""
		number_words = len(words_set)
		zero_list = [0] * number_words  
		words_vector = dict(zip(words_set, zero_list))
		tf = {}
		for i, doc in enumerate(corpus):
			words = doc.split(" ")
			tf[i] = copy.deepcopy(words_vector)
			for word in words:
				tf[i][word] = tf[i][word] + (1/len(words))
		return tf

	def create_idf(self, words_set: set[str], corpus: list[str]) -> dict:
		"""
		Create the inverse document frequency 

		Args: 
			words_set (set[str]): the set of unique words
			corpus (list[str]): the corpus containing all the "documents".
		Returns: 
			idf (dict): dict with word as key and value of 
						log(total # of docs in corpus / # of docs in corpus containing term)
		"""
		idf = {}
		for word in words_set:
			num_docs_w_word = 0
			for i, doc in enumerate(corpus):
				if word in doc.split():
					num_docs_w_word += 1
			idf[word] = math.log10(len(corpus)/num_docs_w_word)
		return idf

	def get_tfidf(self, words_set: set[str], corpus: list[str], tf: dict, idf: dict) -> dict:
		"""
		Create the tf x idf 

		Args: 
			words_set (set[str]): the set of unique words
			corpus (list[str]): the corpus containing all the "documents".
			tf (dict):  the term frequency dict
			idf (dict): the inverse document frequency dict
		Returns: 
			tf_idf (dict): Which is the score that reflects the importance of a term for a document in the corpus.
		"""
		tf_idf = copy.deepcopy(tf)
		for word in words_set:
			for i, doc in enumerate(corpus):
				tf_idf[i][word] = tf[i][word] * idf[word]
		return tf_idf


if __name__ == "__main__":
	test_corpus = ["This here is doc1", "and this here is another doc called doc2"]
	tfidf_test = TFIDFTest()
	words_set = tfidf_test.create_words_set(corpus=test_corpus)
	tf = tfidf_test.create_tf(words_set=words_set, corpus=test_corpus)
	idf = tfidf_test.create_idf(words_set=words_set, corpus=test_corpus)
	tf_idf = tfidf_test.get_tfidf(words_set=words_set, corpus=test_corpus, tf=tf, idf=idf)
	print(f"this here is the term frequence: {tf} \n")
	print(f"this here is the inverse document frequency: {idf} \n")
	print(f"this here is the tfidf:  {tf_idf} \n Thanks a lot!")
	# Test to make sure that we are creating the right output.
	assert(tf != tf_idf)