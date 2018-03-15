import wikipedia
import spelling
import numpy as np
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer


def process_question(question):
    # process the question

    question = question.lower()
    question = question.strip('?').split()

    # check for any misspelled words
    for word in question:
        if spelling.correction(word) != word:
            print("Would you like to replace " + word + " with " + spelling.correction(word) + "? (Y/N)")
            answer = input(">> ").lower()
            if answer == 'y' or answer == 'yes':
                question = [q.replace(word, spelling.correction(word)) for q in question]
            elif answer == 'n' or answer == 'no':
                continue
            else:
                print("Please Enter Yes or No.")

    question = " ".join(question)

    # extract keywords from question
    r = Rake()
    r.extract_keywords_from_text(question)
    return r.get_ranked_phrases()


def documents_retriever(keywords):
    # retrieve 5 most relevant articles from Wikipedia

    keywords_string = ""
    for keyword in keywords:
        keywords_string = keywords_string + ' ' + keyword
        
    names = wikipedia.search(keywords_string, results=5, suggestion=False)
 
    summaries = []
    for name in names:
        summaries.append(wikipedia.summary(name, sentences = 10))

    return summaries


def tf_idf_score(question, summaries):
    # calculate the similarity between retrieved documents and the question 

    summaries.insert(0, question)
    vect = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2), smooth_idf = False)
    tfidf = vect.fit_transform(summaries)
    matrix = (tfidf * tfidf.T).A
    return np.argmax(matrix[0][1:])
    
    
def get_document(question):
    # retrieve the most relevant document

    try:
        keywords = process_question(question)
        summaries = documents_retriever(keywords)
        index = tf_idf_score(question, summaries) + 1
        return summaries[index]
    except:
        return "No answer"
        

