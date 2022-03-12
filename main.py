import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df = pd.read_csv("emails.csv")

arr = df.to_numpy()

types = df['spam'].values

text = df['text'].values

X_train, X_test, y_train, y_test = train_test_split(text, types, test_size=0.2)

array_ham = np.array([X_train[i] for i in range(len(X_train)) if y_train[i] == 0])
array_spam = np.array([X_train[i] for i in range(len(X_train)) if y_train[i] == 1])

ham_mails = np.array([text[i] for i in range(len(text)) if types[i] == 0])
spam_mails = np.array([text[i] for i in range(len(text)) if types[i] == 1])
    
flatten_ham_og = ham_mails.flatten()
flatten_spam_og = spam_mails.flatten()

flatten_ham = array_ham.flatten()
flatten_spam = array_spam.flatten()
flatten_X_test = X_test.flatten()

def understanding_data():
       
    vectorizer_ham = CountVectorizer()
    X_ham = vectorizer_ham.fit_transform(flatten_ham_og)
        
    vectorizer_spam = CountVectorizer()
    X_spam = vectorizer_spam.fit_transform(flatten_spam_og)
    
    ham_feature_names = vectorizer_ham.get_feature_names()
    spam_feature_names = vectorizer_spam.get_feature_names()
        
    ham_feature_freq = X_ham.toarray().sum(axis=0)
    spam_feature_freq = X_spam.toarray().sum(axis=0)
        
    ham_dictionary = dict(zip(ham_feature_names, ham_feature_freq))
    key_list_ham = list(ham_dictionary.keys())
    value_list_ham = list(ham_dictionary.values())
    
    spam_dictionary = dict(zip(spam_feature_names, spam_feature_freq))
    key_list_spam = list(spam_dictionary.keys())
    value_list_spam = list(spam_dictionary.values())
    
    tfidf_transformer_ham = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_transformer_ham.fit(X_ham)
        
    tfidf_transformer_spam = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_transformer_spam.fit(X_spam)
    
    ham_dictionary_tfidf = dict(zip(ham_feature_names, tfidf_transformer_ham.idf_))
    key_list_ham_tfidf = list(ham_dictionary_tfidf.keys())
    value_list_ham_tfidf = list(ham_dictionary_tfidf.values())
    
    spam_dictionary_tfidf = dict(zip(spam_feature_names, tfidf_transformer_spam.idf_))
    key_list_spam_tfidf = list(spam_dictionary_tfidf.keys())
    value_list_spam_tfidf = list(spam_dictionary_tfidf.values())
       
    ham_keys = list(ham_dictionary.keys())
    spam_keys = list(spam_dictionary.keys())
    
    ham_values = list(ham_dictionary.values())
    spam_values = list(spam_dictionary.values())
    
    ham_values_tfidf = list(ham_dictionary_tfidf.values())
    spam_values_tfidf = list(spam_dictionary_tfidf.values())
      
    # Part 1 and Part 3.a 
    for i in range(len(ham_dictionary)):
        
        if ham_keys[i] not in spam_keys and ham_values[i] > 35:
     
            print("ham_keys word -> ",ham_keys[i],": ",ham_values[i])
            print("tfidf value -> ",ham_values_tfidf[i])
    
    print("**********************************************")
    
    for i in range(len(spam_dictionary)):
        
        if spam_keys[i] not in ham_keys and spam_values[i] > 35:
     
            print("spam_keys word -> ",spam_keys[i],": ",spam_values[i])
            print("tfidf value -> ",spam_values_tfidf[i])
    
    
    # Stop-word analysis   
    for word in ['cant', 'nowhere', 'myself', 'hereby', 'please', 'anyhow', 'whoever', 'our', 'last', 'thereupon', 'then', 'yours', 'beside', 'a', 'full', 'been', 'also', 'detail', 'empty', 'off', 'seemed', 'was', 'whereafter', 'me', 'fill', 'never', 'interest', 'once', 'thin', 'give', 'wherein', 'thus', 'ourselves', 'now', 'same', 'nine', 'mostly', 'already', 'am', 'someone', 'something', 'mine', 'another', 'find', 'would', 'become', 'less', 'only', 'hereupon', 'next', 'always', 'might', 'do', 'had', 'amount', 'de', 'latter', 'per', 'seem', 'were', 'while', 'anywhere', 'nor', 'not', 'ours', 'several', 'must', 'with', 'anything', 'sometimes', 'within', 'five', 'alone', 'found', 'here', 'cry', 'third', 'therein', 'take', 'behind', 'hereafter', 'system', 'somewhere', 'others', 'moreover', 'bill', 'for', 'least', 'however', 'put', 'between', 'indeed', 'he', 'them', 'than', 'few', 'somehow', 'becomes', 'she', 'it', 'seeming', 'onto', 'hers', 'to', 'toward', 'himself', 'became', 'own', 'out', 'one', 'whence', 'go', 'everyone', 'may', 'you', 'whereby', 'nothing', 'have', 'over', 'some', 'be', 'towards', 'whenever', 'often', 'around', 'former', 'whose', 'anyway', 'ever', 'has', 'any', 'herein', 'though', 'whatever', 'un', 'thereby', 'until', 'whom', 'beyond', 'further', 'becoming', 'describe', 'fire', 'eight', 'hasnt', 'being', 'whereupon', 'in', 'could', 'meanwhile', 'each', 'even', 'their', 'latterly', 'we', 'forty', 'two', 'move', 'six', 'still', 'yourself', 'perhaps', 'up', 'after', 'during', 'formerly', 'amongst', 'her', 'eleven', 'sincere', 'throughout', 'my', 'if', 'four', 'those', 'bottom', 'whether', 'done', 'rather', 'every', 'the', 'name', 'seems', 'from', 'without', 'inc', 'co', 'more', 'its', 'before', 'etc', 'or', 'above', 'all', 'everything', 'either', 'when', 'twenty', 'us', 'else', 'beforehand', 'no', 'hundred', 'him', 'mill', 'top', 'whither', 'very', 'show', 'why', 'your', 'hence', 'under', 'see', 'back', 'side', 'together', 'wherever', 'what', 'through', 'fifteen', 'thru', 'too', 'how', 'such', 'eg', 'they', 'thereafter', 'thick', 'yet', 'against', 'there', 'well', 'of', 'i', 'this', 'first', 'both', 'since', 'part', 'an', 'none', 'namely', 'almost', 'elsewhere', 'although', 'at', 'noone', 'therefore', 'below', 'enough', 'about', 'ltd', 'by', 'down', 'whole', 'because', 'ie', 'should', 're', 'which', 'fifty', 'who', 'front', 'so', 'call', 'otherwise', 'into', 'these', 'nobody', 'via', 'besides', 'herself', 'most', 'itself', 'where', 'get', 'and', 'other', 'sixty', 'along', 'can', 'con', 'upon', 'themselves', 'yourselves', 'his', 'will', 'due', 'but', 'nevertheless', 'twelve', 'amoungst', 'many', 'neither', 'are', 'across', 'ten', 'as', 'again', 'on', 'whereas', 'afterwards', 'anyone', 'thence', 'everywhere', 'much', 'is', 'serious', 'among', 'that', 'cannot', 'three', 'couldnt', 'sometime', 'keep', 'except', 'made']:
        
        if word in list(ham_dictionary.keys()):
            print("ham dictionary -> ",word,": ", ham_dictionary[word])
        
        if word in list(spam_dictionary.keys()):
            print("spam dictionary -> ",word,": ",spam_dictionary[word])
        
        if word in list(ham_dictionary_tfidf.keys()):
            print("ham dictionary tfidf -> ",word,": ",ham_dictionary_tfidf[word])
        
        if word in list(spam_dictionary_tfidf.keys()):
        
            print("spam dictionary tfidf -> ",word,": ",spam_dictionary_tfidf[word])
        
        print("*******************************************\n")
   
      
# Call the function by removing comment tags
# understanding_data()

def naive_bayes_predict(ngram, min_df_value,tfidf,stopwords):
    
    predictions = []
    
    if (stopwords):
        vectorizer_ham = CountVectorizer(ngram_range = ngram,min_df=min_df_value,stop_words=ENGLISH_STOP_WORDS)
        X_ham = vectorizer_ham.fit_transform(flatten_ham)
            
        vectorizer_spam = CountVectorizer(ngram_range = ngram,min_df=min_df_value,stop_words=ENGLISH_STOP_WORDS)
        X_spam = vectorizer_spam.fit_transform(flatten_spam)
    
    
    else:
    
        vectorizer_ham = CountVectorizer(ngram_range = ngram,min_df=min_df_value)
        X_ham = vectorizer_ham.fit_transform(flatten_ham)
        
        vectorizer_spam = CountVectorizer(ngram_range = ngram,min_df=min_df_value)
        X_spam = vectorizer_spam.fit_transform(flatten_spam)
    
    ham_feature_names = vectorizer_ham.get_feature_names()
    spam_feature_names = vectorizer_spam.get_feature_names()
    
    ham_feature_numbers = len(ham_feature_names)
    spam_feature_numbers = len(spam_feature_names)
    
    
    if tfidf == False:
         
        ham_feature_freq = X_ham.toarray().sum(axis=0)
        spam_feature_freq = X_spam.toarray().sum(axis=0)
        
        
        ham_dictionary = dict(zip(ham_feature_names, ham_feature_freq))
        spam_dictionary = dict(zip(spam_feature_names, spam_feature_freq))
        
        total_ham_feature_freq = ham_feature_freq.sum()
        total_spam_feature_freq = spam_feature_freq.sum()
    
    else:
                
        
        tfidf_transformer_ham = TfidfTransformer(smooth_idf = True, use_idf = True)
        tfidf_transformer_ham.fit(X_ham)
        
        tfidf_transformer_spam = TfidfTransformer(smooth_idf = True, use_idf = True)
        tfidf_transformer_spam.fit(X_spam)
    
        ham_dictionary = dict(zip(ham_feature_names, tfidf_transformer_ham.idf_))
        spam_dictionary = dict(zip(spam_feature_names, tfidf_transformer_spam.idf_))
            
              
        ham_dictionary = {k: v for k, v in sorted(ham_dictionary.items(), key=lambda item: item[1]) if v > 1.5}
        spam_dictionary = {k: v for k, v in sorted(spam_dictionary.items(), key=lambda item: item[1]) if v > 1.5}
        
        
        total_ham_feature_freq = sum(tfidf_transformer_ham.idf_)
        total_spam_feature_freq = sum(tfidf_transformer_spam.idf_)
    
   
   

    vectorizer_X_test = CountVectorizer(ngram_range=ngram)
    X_test = vectorizer_X_test.fit_transform(flatten_X_test)
    X_test_freq = X_test.toarray()
    X_test_feature_names = vectorizer_X_test.get_feature_names()
    
    for i in range(len(X_test_freq)):
    
        naive_bayes_ham_total = math.log((len(array_ham) / len(X_train)),2)
        naive_bayes_spam_total = math.log((len(array_spam) / len(X_train)),2)
        ham_total = 0
        spam_total = 0
        
        for k in range(len(X_test_freq[i])):
            
            X_test_in_ham = X_test_feature_names[k] in ham_dictionary.keys() 
            X_test_in_spam = X_test_feature_names[k] in spam_dictionary.keys()
          
            if X_test_freq[i][k] > 0:
                if X_test_in_ham or X_test_in_spam:
                    
                    if X_test_in_ham:
                        ham_total += X_test_freq[i][k] * math.log((ham_dictionary[X_test_feature_names[k]] + 1) / (total_ham_feature_freq + ham_feature_numbers),2)

                    if X_test_in_spam:
                        spam_total += X_test_freq[i][k] * math.log((spam_dictionary[X_test_feature_names[k]] + 1) / (total_spam_feature_freq + spam_feature_numbers),2) 

                else:
                    
                    if not X_test_in_ham:
                        ham_total += X_test_freq[i][k] * math.log(1 / (total_ham_feature_freq + ham_feature_numbers),2)

                    if not X_test_in_spam:
                        spam_total += X_test_freq[i][k] * math.log(1 / (total_spam_feature_freq + spam_feature_numbers),2) 

        naive_bayes_ham_total -= ham_total
        naive_bayes_spam_total -= spam_total
        
        if naive_bayes_ham_total > naive_bayes_spam_total:
            predictions.append(0)
    
        else:
            predictions.append(1)

        
    return predictions

# to do naive bayes classification using unigrams, one must call the method as below
# naive_bayes_predict((1,1),0.003,False,False)

# to do naive bayes classification using bigrams, one must call the method as below
# naive_bayes_predict((2,2),0.003,False,False)

# to do naive bayes classification using unigrams with tf-idf transformation, one must call the method as below
# naive_bayes_predict((1,1),0.003,True,False)

# to do naive bayes classification using unigrams with extracting stopwords, one must call the method as below
# naive_bayes_predict((1,1),0.003,False,True)

def performance_metrics(predictions):
    
    conf_matrix = metrics.confusion_matrix(y_test,predictions,labels=[0,1])
    
    precision = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    
    recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
    
    f1_score = (2 * recall * precision) / (recall + precision)
    
    accuracy = metrics.accuracy_score(y_test,predictions)
    
    print("ACCURACY: ",accuracy)
    print("PRECISION: ",precision)
    print("RECALL: ",recall)
    print("F1 SCORE: ",f1_score)
    

# Main
print("NAIVE BAYES CLASSIFICATION USING UNIGRAMS:")
performance_metrics(naive_bayes_predict((1,1),0.003,False,False))

print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING BIGRAMS:")
performance_metrics(naive_bayes_predict((2,2),0.003,False,False))

print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING UNIGRAMS WITH TF-IDF TRANSFORMATION:")
performance_metrics(naive_bayes_predict((1,1),0.003,True,False))

print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING BIGRAMS WITH TF-IDF TRANSFORMATION:")
performance_metrics(naive_bayes_predict((2,2),0.003,True,False))

print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING UNIGRAMS AND REMOVING STOPWORDS:")
performance_metrics(naive_bayes_predict((1,1),0.003,False,True))

print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING BIGRAMS AND REMOVING STOPWORDS:")
performance_metrics(naive_bayes_predict((2,2),0.003,False,True))


print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING UNIGRAMS AND REMOVING STOPWORDS AND USING TF-IDF TRANSFORMATION:")
performance_metrics(naive_bayes_predict((1,1),0.003,True,True))


print("\n\n")
print("NAIVE BAYES CLASSIFICATION USING BIGRAMS AND REMOVING STOPWORDS AND USING TF-IDF TRANSFORMATION:")
performance_metrics(naive_bayes_predict((2,2),0.003,True,True))
