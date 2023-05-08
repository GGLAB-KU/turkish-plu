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

def tf_idf_filter(step, step_list):
  vect1 = TfidfVectorizer()
  step_tfidf_matrix = vect1.fit_transform([step])
  
  doc = [" ".join([step for step in step_list])]
  vect2 = TfidfVectorizer()
  doc_tfidf_matrix = vect2.fit_transform(doc)
  df = pd.DataFrame(doc_tfidf_matrix.toarray(), columns = vect2.get_feature_names())
  
  try:
    tfidf_scores = [df[token][0] for token in vect1.get_feature_names()]
    if max(tfidf_scores) >= 0.25:
      return True
  except ValueError:
    return False

  return False

def lexical_overlap_filter(positive_candidate, negative_candidates):
  positive_candidate = set([token.lemma for token in nlpturk(" ".join(word for word in positive_candidate.split() if word.lower() not in stop_words))])
  negative_candidates = [set([token.lemma for token in nlpturk(" ".join(word for word in negative_candidate.split() if word.lower() not in stop_words))]) for negative_candidate in negative_candidates]
  if max([len(negative_candidate.intersection(positive_candidate)) / len(negative_candidate.union(positive_candidate)) for negative_candidate in negative_candidates]) >= 0.5:
    return False

  return True

def length_filter(text, tokenizer, threshold):
  return len(tokenizer.tokenize(text)) > threshold

def similarity_filter_for_next_event_prediction(positive_candidate, negative_candidates, simcse):
    similarities = simcse.similarity([positive_candidate], negative_candidates).tolist()[0]
    if (max(similarities) >= 0.775):
      return False
        
    return True

def similarity_filter_for_step_inference(goal, negative_candidates, wikihow, simcse):
    all_steps_for_goal = [article_dict["caption"] for article_dict in wikihow if (article_dict["task"] == goal)]
    all_steps_for_goal = [step for method in all_steps_for_goal for step in method]

    for negative_candidate in negative_candidates:
      similarities = simcse.similarity([negative_candidate], all_steps_for_goal).tolist()[0]
      if (max(similarities) >= 0.775):
        return False
          
    return True

def similarity_filter_for_goal_inference(step, negative_candidates, wikihow, simcse):
    for negative_candidate in negative_candidates:
      all_steps_for_negative_candidate = [article_dict["caption"] for article_dict in wikihow if (article_dict["task"] == negative_candidate)]
      all_steps_for_negative_candidate = [step for method in all_steps_for_negative_candidate for step in method]

      similarities = simcse.similarity([step], all_steps_for_negative_candidate).tolist()[0]
      if (max(similarities) >= 0.775):
        return False
          
    return True