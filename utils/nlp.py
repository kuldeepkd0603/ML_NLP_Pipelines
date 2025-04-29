import re
import joblib
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from utils.constant import nlp_constants
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")


label_mapping = {
            0: "Politics",
            1: "Sport",
            2: "Technology",
            3: "Entertainment",
            4: "Business"}

def clean_text1(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower()


def clean_text(text):
    try:
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text
    except Exception as e:
        return f"error occured in clean text function of nlp moulde {e}"
    
def get_sentence_vector(tokens):
    try:
        try:
            w2v_model = Word2Vec.load(nlp_constants.get("word2vec-vectorizer"))
        except Exception as e:
            return f"word2vec model not found {e}"
        word_vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(100)
    except Exception as e:
        return "error occured in get_sentance_vector func {e}"
    
 
def predict_word2vec(text):
    try:        
        try:
            model=joblib.load(nlp_constants.get("word2vec-ml-model"))
        except Exception as e:
                return f"word2vec-ml-model no found {e}"
            
        cleaned = clean_text1(text)
        tokens = word_tokenize(cleaned)
        vector = get_sentence_vector(tokens)
        if vector.ndim == 1:
                vector = vector.reshape(1, -1)
        elif vector.shape[1] != 100:
                return "Input vector shape mismatch. Expected (1, 100)."
        
        predicted = model.predict(vector)[0]
        return predicted
    except Exception as e:
        return "error occured in predict word2vec {e}"
 
    
def predict_with_ner(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        category = predict_word2vec(text)
        category=label_mapping.get(category, "unknown")
        doc = nlp(text)
    
        entities = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]
       
        return category, entities
    except Exception as e:
        return f"error occured in predict with ner {e}"

def nlp(nlp_model,text):
    try:
        
        if nlp_model=="tfidf":
            try:
                scaler=joblib.load(nlp_constants.get("tf-idf-scaler"))
            except Exception as e:
                return f"tfidf model scaler not found {e}"
            
            try:
                vectorizer=joblib.load(nlp_constants.get("tf-idf-vector"))
            except Exception as e:
                return f"tfidf model vectorizer  not found {e}"
            
            try:
                model=joblib.load(nlp_constants.get("tf-idf-model"))
            except Exception as e:
                return f"tfidf model  not found {e}"
            
            text=clean_text(text)
            text_vector = vectorizer.transform([text])
            scaled_input = scaler.transform(text_vector.toarray())
            predicted_class = model.predict(scaled_input)[0]
            return label_mapping.get(predicted_class, "unknown")
            
            
        elif (nlp_model=="word2vec"):
               predicted_class=predict_word2vec(text)
               return label_mapping.get(predicted_class, "unknown")
            
        elif (nlp_model=="NER"):
            category, entities = predict_with_ner(text)
            final_output = f"<h3>Predicted Category:</h3> {category}<br><br>"
            if entities:
                 final_output += "<h3>Named Entities Found:</h3><ul>"
                 for ent_text, ent_label, ent_explain in entities:
                      final_output += f"<li>{ent_text} ({ent_label}): {ent_explain}</li>"
                 final_output += "</ul>"
            else:
               final_output += "<h3>No named entities detected.</h3>"
            return final_output
        else:
            return "wrong model choiice"
        
    except Exception as e:
        return f"error occured in nlp function of npl module {e}"
        
