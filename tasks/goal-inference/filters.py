from transformers import BertModel, BertTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nlpturk
import torch
from sentence_transformers import util

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('turkish'))

def tf_idf_filter(step, step2goal, wikihow):
  goal = step2goal[step]
  all_steps = [article_dict["caption"] for article_dict in wikihow if (article_dict["task"] == goal)]
  doc = [" ".join([step for method in all_steps for step in method])]
  vect = TfidfVectorizer()
  tfidf_matrix = vect.fit_transform(doc)
  df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
  
  try:
    tfidf_scores = [df[token][0] for token in vect.get_feature_names()]
    if max(tfidf_scores) >= 0.2:
      return True
  except ValueError:
    return False

  return False

def lexical_overlap_filter(positive_candidate, negative_candidates):
  positive_candidate = set([token.lemma for token in nlpturk(" ".join(word for word in positive_candidate.split() if word.lower() not in stop_words))])

  for negative_candidate in negative_candidates:
    negative_candidate = set([token.lemma for token in nlpturk(" ".join(word for word in negative_candidate.split() if word.lower() not in stop_words))])
    if len(negative_candidate.intersection(positive_candidate)) / len(negative_candidate.union(positive_candidate)) >= 0.5:
      return False

  return True

def length_filter(text, tokenizer, threshold):
  return len(tokenizer.tokenize(text)) > 7

def similarity_filter_for_step_inference(goal, negative_candidates, wikihow, simcse):
    all_steps_for_goal = [article_dict["caption"] for article_dict in wikihow if (article_dict["task"] == goal)]
    all_steps_for_goal = [step for method in all_steps_for_goal for step in method]

    for negative_candidate in negative_candidates:
      similarities = simcse.similarity([negative_candidate], all_steps_for_goal).tolist()[0]
      if (max(similarities) >= 0.8):
        return False
          
    return True

def similarity_filter_for_goal_inference(step, negative_candidates, wikihow, simcse):
    for negative_candidate in negative_candidates:
      all_steps_for_negative_candidate = [article_dict["caption"] for article_dict in wikihow if (article_dict["task"] == negative_candidate)]
      all_steps_for_negative_candidate = [step for method in all_steps_for_negative_candidate for step in method]

      similarities = simcse.similarity([step], all_steps_for_negative_candidate).tolist()[0]
      if (max(similarities) >= 0.8):
        return False
          
    return True