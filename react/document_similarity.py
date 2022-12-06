import pickle
from pathlib import Path
import pandas as pd
import re
import text_normalizer as tn
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np

stopword_list = nltk.corpus.stopwords.words('english')

classification_pipeline = pickle.load(open('classification_pipeline.pickle', 'rb'))

# Function for new document analysis in the classification step

def open_predict_classification(file):
    new_document = []
    with open(file, encoding='latin-1') as f:
        lines = f.readlines()
        filename = re.sub('\/Users\/apple\/Documents\/IU\/NLP\/FinalProject\/Data\/SpamAssassin-master\/\w+\/', '', str(file))
        body = ' '.join([l.strip() for l in lines[:]])
        subject = [re.findall(r'(?<=Subject:\s).*', l) for l in lines[:] if l.startswith('Subject') is True]
        to_email = [re.findall(r'(?<=To:\s).*', l) for l in lines[:] if l.startswith('To:') is True]
        #to email = [re.findall(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+', email) for email in to_email]
        from_email = [re.findall(r'(?<=From:\s).*', l) for l in lines[:] if l.startswith('From:') is True]
        new_document.append([filename, body, subject, to_email, from_email])
        
    document = pd.DataFrame(new_document, columns=['filename', 'body', 'subject', 'to_email', 'from_email'])
    
    ''' print("Document:\n")
    print("Filename: " + str(document['filename'][0]) + "...")
    print("Body: " + document['body'][0][:100])
    print("Subject: " + str(document['subject'][0][0][0]))
    print("Recipients: " + str(document['to_email'][0]))
    print("Sender: " + str(document['from_email'][0]))
    print("-----------------\n") '''
    
    normalized_doc = tn.normalize_corpus(corpus=document['body'], html_stripping=True, contraction_expansion=True, 
                                  accented_char_removal=True, text_lower_case=True, text_lemmatization=True, 
                                  text_stemming=False, special_char_removal=True, remove_digits=False,
                                  stopword_removal=True, stopwords=stopword_list)
    
    document['cleaned_body'] = normalized_doc
    
    ''' print("New document cleaned body:\n")
    print(str(document['cleaned_body'][0][:100]) + "...")
    print("\n-----------------\n") '''
    
    predicted_likelihood = str(float(classification_pipeline.predict_proba(document['cleaned_body']).max(axis=1)[0]) * 100)
    predicted_classification = str(classification_pipeline.predict(document['cleaned_body'])[0])
    
    response = "Likelihood\n" + predicted_classification + "\n" +  predicted_likelihood[:5] + "%"
    return response
    #print(predicted_classification)
    #print(predicted_likelihood[:5] + "%")



def process_email_similarity(email):
    current_directory =  os.path.abspath(os.curdir)
    doc = []
    csv_path = current_directory + "/docs.csv"
    with open(csv_path, encoding='latin-1') as f:
        df_document = pd.read_csv(f, index_col=0)
    
    with open(email, encoding='latin-1') as f:
        lines = f.readlines()
        label = ''
        label_num = ''
        cleaned_body = ''
        filename = str(email)
        body = ' '.join([l.strip() for l in lines[:]])
        for l in lines[:]:
            if l.startswith('Subject') is True:
                subject = re.sub(r'Subject:\s|\n', '', l)
            if l.startswith('To:') is True:
                to_email = re.sub(r'To:\s|\n', '', l)
            if l.startswith('From:') is True:
                from_email = re.sub(r'From:\s|\n', '', l)
        #subject = re.sub(r'\\n', '', subject)
        #to_email = [re.findall(r'(?<=To:\s).*', l) for l in lines[:] if l.startswith('To:') is True]
        #from_email = [re.findall(r'(?<=From:\s).*', l) for l in lines[:] if l.startswith('From:') is True]
        doc.append([filename, body, subject, to_email, from_email, label, label_num, cleaned_body])
        doc = pd.DataFrame(doc, columns=['filename', 'body', 'subject', 'to_email', 'from_email', 'label', 'label_num', 'cleaned_body'])
        df_document = pd.concat([df_document, doc], axis=0, ignore_index=True)
    
        df_to_clean = df_document.loc[df_document['subject'] == subject]
    
    normalized_doc = tn.normalize_corpus(corpus=df_to_clean['body'], html_stripping=True, contraction_expansion=True, 
                                  accented_char_removal=True, text_lower_case=True, text_lemmatization=True, 
                                  text_stemming=False, special_char_removal=True, remove_digits=False,
                                  stopword_removal=True, stopwords=stopword_list)
    
    df_document.loc[df_document['subject'] == subject, 'cleaned_body'] = normalized_doc
    
    df_document.fillna('', inplace=True)
  
    tf_similarity = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tf_similarity.fit_transform(df_document['cleaned_body'])

    doc_sim = cosine_similarity(tfidf_matrix)
    doc_sim_df = pd.DataFrame(doc_sim)
    doc_sim_df.head()
    
    document_list = df_document['subject'].values

    # find email id
    email_idx = np.where(document_list == subject)[0][0]
    
    # get email similarities
    email_similarities = doc_sim_df.iloc[email_idx].values
    
    # get top 10 similar email IDs
    similar_email_idxs = np.argsort(-email_similarities)[1:11]
    
    # get top 10 documents
    similar_emails = document_list[similar_email_idxs]
    
    # return the top 10 documents
    response = ''
    for x in range(len(similar_emails)-1):
        response += "Subject: " + str(similar_emails[x]) + " |----| " + "Cosine Similarity: " + str(email_similarities[x]) + "\n"

    return response



#open_predict_classification("/Users/apple/Desktop/test.txt")
#process_email_similarity("/Users/apple/Desktop/test.txt")