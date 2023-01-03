import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import joblib

# set parameters
n_topics = 100
additional_stop_words = [

]

# create file root
file_root = "C:\\GitHub\\political_advertisements"
pd.options.mode.chained_assignment = None

# read data
data_intake = pd.read_csv(f"{file_root}\\intake\\fbpac-ads-en-US.csv")
data_filter = data_intake[["id", "message", "created_at"]]

# clean the data
def clean_text(text):
    clean_numbers = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)
    clean_sites = re.sub(r"https?://\S+|www\.\S+", "", clean_numbers)
    cleaned_html = re.sub("<[^<]+?>", "", clean_sites)
    cleaned_text = cleaned_html.replace("href", "")
    return cleaned_text

data_filter["message_clean"] = data_filter["message"].apply(clean_text)

# add to stop words
stop_words_edit = sk.feature_extraction.text.ENGLISH_STOP_WORDS.union(additional_stop_words)
list_stop_words_edit = [term for term in stop_words_edit]

# vectorize the text
count_vect = CountVectorizer(max_df = 0.5, min_df = 2, stop_words = list_stop_words_edit)
doc_term_matrix = count_vect.fit_transform(data_filter["message_clean"].values.astype("U"))
joblib.dump(count_vect, f"{file_root}\\working\\text_vectorizer.bin")

# use LDA to model topics
LDA = LatentDirichletAllocation(n_components = n_topics, random_state = 42)
LDA.fit(doc_term_matrix)
joblib.dump(LDA, f"{file_root}\\working\\model_LDA.bin")

# retrieve the topics and terms associated
print("\nNOTE: Printing topics:")
for topic in range(n_topics):
    topic_selected = LDA.components_[topic]
    top_topic_words = topic_selected.argsort()[-5:]
    topic_terms = []
    for term in top_topic_words:
        topic_terms.append(count_vect.get_feature_names_out()[term])
    print(f"  Topic {topic}: {topic_terms}")

# assign topics to documents
topic_values = LDA.transform(doc_term_matrix)
data_assigned = data_filter
data_assigned["topic"] = topic_values.argmax(axis = 1)
data_assigned.to_csv(f"{file_root}\\working\\ads_assigned_topic.csv")