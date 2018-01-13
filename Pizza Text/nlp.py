from nltk.corpus import brown
# print(brown.words())
# print(brown.fileids())
# print(brown.raw('cr08').strip()[:1000])

from nltk.corpus import webtext
import re
# print(webtext.fileids())

# Each line is one advertisement.
# for i, line in enumerate(webtext.raw('singles.txt').split('\n')):
#     if i > 10: # Lets take a look at the first 10 ads.
#         break
#     print(str(i) + ':\t' + line)

import pandas as pd
single_no8 = webtext.raw('singles.txt').split('\n')[8]
# print(single_no8)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# Sentence Tokenization
from nltk import sent_tokenize, word_tokenize
# print(sent_tokenize(single_no8))

# for sent in sent_tokenize(single_no8):
#     print(word_tokenize(sent))

# for sent in sent_tokenize(single_no8):
    # It's a little in efficient to loop through each word,
    # after but sometimes it helps to get better tokens.
    # print([word.lower() for word in word_tokenize(sent)])
    # Alternatively:
    # print(list(map(str.lower, word_tokenize(sent))))

from nltk.corpus import stopwords

single_no8_tokenized_lowered = list(map(str.lower, word_tokenize(single_no8)))
stopwords_en = set(stopwords.words('english'))
# List comprehension.
# print([word for word in single_no8_tokenized_lowered if word not in stopwords_en])

from string import punctuation
# It's a string so we have to them into a set type
# print('From string.punctuation:', type(punctuation), punctuation)
stopwords_json = {"en":["a","request","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
# Combine the stopwords. Its a lot longer so I'm not printing it out...
stopwords_json_en = set(stopwords_json['en'])
stopwords_nltk_en = set(stopwords.words('english'))
stopwords_punct = set(punctuation)
# Combine the stopwords. Its a lot longer so I'm not printing it out...
stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)
# print(stopwords_en_withpunct)
# print([word for word in single_no8_tokenized_lowered if word not in stopwords_en_withpunct])

from nltk.stem import PorterStemmer
porter = PorterStemmer()

# for word in ['walking', 'walks', 'walked']:
#     print(porter.stem(word))

# import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

# for word in ['walking', 'walks', 'walked']:
#     print(wnl.lemmatize(word))
#
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
#
# wnl = WordNetLemmatizer()
# #
# # import nltk
# # nltk.download('averaged_perceptron_tagger')
#
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

#
def lemmatize_sent(text):
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(word_tokenize(text))]


# print(lemmatize_sent('He is walking to school'))#
def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text)
            if word not in stoplist_combined
            and word.isalpha()]
# # print(single_no8)
# # print(preprocess_text(single_no8))
#
# from collections import Counter
#
# sent1 = "The quick brown fox jumps over the lazy brown dog."
# sent2 = "Mr brown jumps over the lazy fox."
#
# # Lemmatize and remove stopwords
# processed_sent1 = preprocess_text(sent1)
# processed_sent2 = preprocess_text(sent2)
# # print('Processed sentence:')
# # print(processed_sent1)
# # print()
# # print('Word counts:')
# # print(Counter(processed_sent1))
# # print('Processed sentence:')
# # print(processed_sent2)
# # print()
# # print('Word counts:')
# # print(Counter(processed_sent2))
#
# from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
#
# sent1 = "The quick brown fox jumps over the lazy brown dog."
# sent2 = "Mr brown jumps over the lazy fox."
#
# with StringIO('\n'.join([sent1, sent2])) as fin:
#     # Override the analyzer totally with our preprocess text
#     count_vect = CountVectorizer(stop_words=stoplist_combined,
#                                  tokenizer=word_tokenize)
#     count_vect.fit_transform(fin)
# print(count_vect.vocabulary_)
#
# # in matrix form
# from operator import itemgetter
#
# # Print the words sorted by their index
# words_sorted_by_index, _ = zip(*sorted(count_vect.vocabulary_.items(), key=itemgetter(1)))
#
# print(preprocess_text(sent1))
# print(preprocess_text(sent2))
# print()
# print('Vocab:', words_sorted_by_index)
# print()
# print('Matrix/Vectors:\n', count_vect.transform([sent1, sent2]).toarray())

import pandas as pd
import json
import csv
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing

import seaborn as sns
import matplotlib.pyplot as plt

with open('data/train.json') as fin:
    trainjson = json.load(fin)
df_train = pd.io.json.json_normalize(trainjson) # Pandas magic...

with open('data/test.json') as fin:
    testjson = json.load(fin)
df_test = pd.io.json.json_normalize(testjson) # Pandas magic...

train_df_fail = df_train[df_train['requester_received_pizza'] == False]
train_df_success = df_train[df_train['requester_received_pizza'] == True]
df_train = pd.concat([train_df_fail, train_df_success,
                      train_df_success, train_df_success], axis = 0)
# print(df_train.head())
# df_train['text']=df_train['request_title']+df_train['request_text_edit_aware']
# print(df['train_text'].head())
# print(df_train['requester_subreddits_at_request'])
# target= df_train['requester_received_pizza']
train_labels = np.ravel(df_train[['requester_received_pizza']])

def create_features(df):
    # Length of the post's title
    df['title_len'] = df.request_title.str.len()
    # g = sns.factorplot(x="title_len", y="requester_received_pizza", data=df)
    # g = g.set_ylabels("Pizza Probability")
    # plt.show()

    # Length of the post's text
    df['request_len'] = df.request_text_edit_aware.str.len()


    # Whether or not the user was subscribed to exactly 0 subreddits
    df['zero_subreddits'] = df['requester_number_of_subreddits_at_request'] == 0

    # Total number of comments + number of random acts of pizza comments
    df['total_comments'] = (df['requester_number_of_comments_in_raop_at_request'] +
                            df['requester_number_of_comments_at_request'])

    # The ratio of the user's total comments to their number of random acts of pizza comments
    df['comment_ratio'] = (df['total_comments'] /
                           df['requester_number_of_comments_in_raop_at_request'])

    df.loc[df['comment_ratio'] == np.inf, 'comment_ratio'] = \
        df.loc[df['comment_ratio'] != np.inf, 'comment_ratio'].mean(skipna=True)

    df.loc[pd.isnull(df['comment_ratio']), 'comment_ratio'] = \
        df.loc[pd.notnull(df['comment_ratio']), 'comment_ratio'].mean(skipna=True)

    # The number of upvotes they've received
    df['upvotes'] = (df['requester_upvotes_minus_downvotes_at_request'] +
                     df['requester_upvotes_plus_downvotes_at_request']) / 2

    # The number of downvotes they've received
    df['downvotes'] = (df['requester_upvotes_plus_downvotes_at_request'] -
                       df['upvotes'])

    # The ratio of upvotes they've received
    df['upvote_ratio'] = (df['upvotes'] /
                          (df['upvotes'] + df['downvotes']))

    df.loc[df['upvote_ratio'] == np.inf, 'upvote_ratio'] = \
        df.loc[df['upvote_ratio'] != np.inf, 'upvote_ratio'].mean(skipna=True)

    df.loc[pd.isnull(df['upvote_ratio']), 'upvote_ratio'] = \
        df.loc[pd.notnull(df['upvote_ratio']), 'upvote_ratio'].mean(skipna=True)

    # Get the date in order to make future variables
    df['date'] = pd.to_datetime(df['unix_timestamp_of_request_utc'], unit='s')

    # Hour of the post
    df['hour'] = pd.DatetimeIndex(df['date']).hour

    # Day of the post
    df['day'] = pd.DatetimeIndex(df['date']).day

    # The post's day of the week
    df['weekday'] = pd.DatetimeIndex(df['date']).weekday

    # Whether the post was made in the first half of the month
    df['first_half_of_month'] = df['day'] <= 15
    df['first_half_of_month'] = df['first_half_of_month'].astype(int)

    # Whether the post was made on a weekend
    df['weekend'] = (df['weekday'] == 5) | (df['weekday'] == 6)
    df['weekend'] = df['weekend'].astype(int)

    # Whether the post was made in the morning
    df['morning'] = (df['hour'] >= 6) & (df['hour'] < 12)
    df['morning'] = df['morning'].astype(int)

    # Whether the post was made in the afternoon
    df['afternoon'] = (df['hour'] >= 12) & (df['hour'] < 16)
    df['afternoon'] = df['afternoon'].astype(int)

    # Whether the post was made in the evening
    df['evening'] = (df['hour'] >= 16) & (df['hour'] < 20)
    df['evening'] = df['evening'].astype(int)

    # Whether the post was made at night
    df['night'] = (df['hour'] >= 20) & (df['hour'] < 23)
    df['night'] = df['night'].astype(int)

    # Whether the post was made late at night
    df['latenight'] = (df['hour'] >= 23) & (df['hour'] < 6)
    df['latenight'] = df['latenight'].astype(int)

    # Whether there was a difference between the utc and unix timestamp
    df['utcdiff'] = (df['unix_timestamp_of_request_utc'] -
                     df['unix_timestamp_of_request'])

    # Month of the post
    df['month'] = pd.DatetimeIndex(df['date']).month

    # Week of the post
    df['week'] = pd.DatetimeIndex(df['date']).week

    # Get the US federal holidays and categorize the request dates as holiday or not.
    cal = USFederalHolidayCalendar()
    holiday_list = cal.holidays(start=df['date'].min(),
                                end=df['date'].max())
    holiday_list = [time.date() for time in holiday_list]

    df['justdate'] = [time.date() for time in df['date']]
    df['holiday'] = df['justdate'].isin(holiday_list)
    df['holiday'] = df['holiday'].astype(int)
    # Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived
    # g = sns.heatmap(df.corr(), annot=True, fmt=".0f", cmap="coolwarm")
    # plt.show()
    return (df)


train_df = create_features(df_train)
test_df = create_features(df_test)

### Code to add in text features
def create_text_features(df):
    # Training data with engineered features
    df_new = df

    # Number of words in title
    df_new['title_nwords'] = df['request_title'].apply(lambda x: len(x.split(' ')))

    # Number of words in body
    df_new['body_nwords'] = df['request_text_edit_aware'].apply(lambda x: len(x.split(' ')))

    # Number of sentences in body
    sen_count = re.compile(r'([a-zA-Z][^\.!?]*[\.!?])', re.M)
    df_new['body_sents'] = df['request_text_edit_aware'].apply(lambda x: len(sen_count.findall(x)))

    # Post includes an image
    image_count = re.compile(r'(imgur\.com|\.jpg)', re.IGNORECASE)
    images = df['request_text_edit_aware'].apply(lambda x: len(image_count.findall(x)))
    df_new['has_image'] = np.where(images > 0, 1, 0)

    # Post includes a 'tl;dr'
    tldr_count = re.compile(r'(tl;dr|tldr|tl,dr|tl:dr)', re.IGNORECASE)
    tldrs = df['request_text_edit_aware'].apply(lambda x: len(tldr_count.findall(x)))
    df_new['has_tldr'] = np.where(tldrs > 0, 1, 0)

    # More than zero all-caps words in title (min char = 3)
    df_new['title_caps'] = df['request_title'].apply( \
        lambda x: np.where(sum(1 for c in re.split('\W+', x) if c.isupper() and len(c) > 3) > 0, 1, 0))

    # More than zero all-caps words in body (min char = 2)
    df_new['body_caps'] = df['request_text_edit_aware'].apply( \
        lambda x: np.where(sum(1 for c in re.split('\W+', x) if c.isupper() and len(c) > 2) > 0, 1, 0))

    # Exclamation marks in title
    excl_marks = re.compile(r'(!)', re.M)
    df_new['excl_marks_title'] = df['request_title'].apply(lambda x: len(excl_marks.findall(x)))
    # print train_df_f['excl_marks']

    # Exclamation marks in body
    df_new['excl_marks_body'] = df['request_text_edit_aware'].apply(lambda x: len(excl_marks.findall(x)))

    # Dollar signs in title
    dollar_signs = re.compile(r'(\$|dollar)', re.IGNORECASE)
    df_new['dollars_title'] = df['request_title'].apply(lambda x: len(dollar_signs.findall(x)))

    # Dollar signs in body
    df_new['dollars_body'] = df['request_text_edit_aware'].apply(lambda x: len(dollar_signs.findall(x)))

    # Categories
    desire = re.compile(
        r'(friend|party|birthday|boyfriend|girlfriend|date|drinks|drunk|wasted|invite|invited|celebrate|celebrating|game|games|movie|beer|crave|craving)',
        re.IGNORECASE)
    family = re.compile(r'(husband|wife|family|parent|parents|mother|father|mom|mum|son|dad|daughter)', re.IGNORECASE)
    job = re.compile(r'(job|unemployment|employment|hire|hired|fired|interview|work|paycheck)', re.IGNORECASE)
    money = re.compile(r'(money|bill|bills|rent|bank|account|paycheck|due|broke|bills|deposit|cash|dollar|dollars|bucks|paid|payed|buy|check|spent|financial|poor|loan|credit|budget|day|now| \
        time|week|until|last|month|tonight|today|next|night|when|tomorrow|first|after|while|before|long|hour|Friday|ago|still|due|past|soon|current|years|never|till|yesterday|morning|evening)',
                       re.IGNORECASE)
    student = re.compile(
        r'(college|student|university|finals|study|studying|class|semester|school|roommate|project|tuition|dorm)',
        re.IGNORECASE)

    df_new['desire_category'] = df['request_text_edit_aware'].apply(lambda x: len(desire.findall(x)))
    df_new['family_category'] = df['request_text_edit_aware'].apply(lambda x: len(family.findall(x)))
    df_new['job_category'] = df['request_text_edit_aware'].apply(lambda x: len(job.findall(x)))
    df_new['money_category'] = df['request_text_edit_aware'].apply(lambda x: len(money.findall(x)))
    df_new['student_category'] = df['request_text_edit_aware'].apply(lambda x: len(student.findall(x)))

    # Gratitude
    gratitude = re.compile(r'(thank|thanks|thankful|appreciate|grateful|gratitude|advance)', re.IGNORECASE)
    df_new['gratitude'] = df['request_text_edit_aware'].apply(lambda x: np.where(len(gratitude.findall(x)) > 0, 1, 0))

    # Reciprocity
    reciprocity = re.compile(
        r'(pay it forward|pay forward|paid it forward|pay the act forward|pay the favor back|paying it forward|pay this forward|pay pizza forward|pay back|pay it back|pay you back|return the favor|return the favour|pay a pizza forward|repay)',
        re.IGNORECASE)
    df_new['reciprocity'] = df['request_text_edit_aware'].apply(
        lambda x: np.where(len(reciprocity.findall(x)) > 0, 1, 0))

    return (df_new)

train_df = create_text_features(train_df)
test_df = create_text_features(test_df)

train_data_categorical = train_df[['hour', 'week', 'day', 'weekday', 'month', 'utcdiff']]
test_data_categorical = test_df[['hour', 'week', 'day', 'weekday', 'month', 'utcdiff']]

train_data = train_df.drop(['request_text', 'requester_received_pizza',
                            'giver_username_if_known', 'post_was_edited',
                            'request_id', 'request_text_edit_aware',
                            'request_title',
                            'requester_subreddits_at_request',
                            'unix_timestamp_of_request_utc',
                            'unix_timestamp_of_request',
                            'requester_username', 'requester_user_flair',
                            'number_of_downvotes_of_request_at_retrieval',
                            'number_of_upvotes_of_request_at_retrieval',
                            'request_number_of_comments_at_retrieval',
                            'requester_account_age_in_days_at_retrieval',
                            'requester_days_since_first_post_on_raop_at_retrieval',
                            'requester_number_of_comments_at_retrieval',
                            'requester_number_of_comments_in_raop_at_retrieval',
                            'requester_number_of_posts_at_retrieval',
                            'requester_number_of_posts_on_raop_at_retrieval',
                            'requester_upvotes_minus_downvotes_at_retrieval',
                            'requester_upvotes_plus_downvotes_at_retrieval',
                            'date', 'hour', 'week', 'day', 'weekday', 'month', 'utcdiff', 'justdate'
                            ],
                           axis = 1)

test_data = test_df.drop(['giver_username_if_known', 'request_id',
                          'request_text_edit_aware',
                            'request_title',
                            'requester_subreddits_at_request',
                            'unix_timestamp_of_request_utc',
                            'unix_timestamp_of_request',
                            'requester_username',
                            'date', 'hour', 'week', 'day', 'weekday', 'month', 'utcdiff', 'justdate'
                            ],
                           axis = 1)

# train_data_dummy = pd.DataFrame(index = train_data.index)
# test_data_dummy = pd.DataFrame(index = test_data.index)
# # print(train_data_dummy.head())
#
# for column in train_data_categorical.columns:
#     train_data_dummy = pd.concat([train_data_dummy,
#                                   pd.get_dummies(train_data_categorical[column],
#                                                 prefix = column)], axis = 1)
#     test_data_dummy = pd.concat([test_data_dummy,
#                                  pd.get_dummies(test_data_categorical[column],
#                                                 prefix = column)], axis = 1)
#     # train_data_dummy = pd.get_dummies(train_data_categorical[column],
#     #                                              prefix = column)
#     # test_data_dummy = pd.get_dummies(test_data_categorical[column],
#     #                                   prefix=column)
#
# train_data = pd.concat([train_data, train_data_dummy], axis = 1)
# test_data = pd.concat([test_data, test_data_dummy], axis = 1)
# print(train_data_dummy.head())

# The code below creates a subreddit frequency dictionary, and also a
# dictionary of lists for each subreddit with 0/1 values for each observation
# in the training and test dataset. This dict of lists is basically the
# dummy coding for each subreddit name, and will be turned into a pandas
# dataframe.
def get_subreddit_user_list_train(df, colname):
    subreddit_freq = {}
    subreddit_user_list = {}
    index = 0
    for subreddit_list in list(df[colname]):
        for subreddit in subreddit_list:
            if subreddit in subreddit_freq:
                subreddit_freq[subreddit] += 1
            else:
                subreddit_freq[subreddit] = 1
                subreddit_user_list[subreddit] = []
            for i in range(len(subreddit_user_list[subreddit]), index):
                subreddit_user_list[subreddit].append(0)
            subreddit_user_list[subreddit].append(1)
        index += 1
    return subreddit_freq, subreddit_user_list

# This function does the same process above, but for the test data. The function is
# different because it will NOT create new subreddit features that don't exist in
# the train data set. Important for proper training.
def get_subreddit_user_list_test(df, colname, subreddit_freq):
    subreddit_user_list = {}
    index = 0

    for subreddit in subreddit_freq:
        subreddit_user_list[subreddit] = []

    for subreddit_list in list(df[colname]):
        for subreddit in subreddit_list:
            if subreddit in subreddit_user_list:
                subreddit_freq[subreddit] += 1
                for i in range(len(subreddit_user_list[subreddit]), index):
                    subreddit_user_list[subreddit].append(0)
                subreddit_user_list[subreddit].append(1)
        index += 1
    return subreddit_freq, subreddit_user_list

# Fill with zeroes to create a full table of values for it to be used in
# the dataframe. In other words, make the matrix dense.
def subreddit_fill_zeroes(subreddit_user_list, size):
    for subreddit in subreddit_user_list:
        for i in range(len(subreddit_user_list[subreddit]), size):
            subreddit_user_list[subreddit].append(0)

# Create the subreddit columns for training data
# Turn into a pandas dataframe based on the training data.
subreddit_freq, subreddit_user_list = get_subreddit_user_list_train(train_df,
                                          'requester_subreddits_at_request')
# print(subreddit_user_list,subreddit_freq)
subreddit_fill_zeroes(subreddit_user_list, train_df.shape[0])

train_subreddit_user_list = pd.DataFrame(subreddit_user_list,
                                         index = range(train_df.shape[0]))

# Use the subreddit frequency table to create the dummy value columns for
# test data. We will only use subreddits that already existed in the train
# data above, and will ignore new subreddits that only exist in test data.
# Turn the result into a pandas dataframe based on test data.
subreddit_freq, subreddit_user_list = get_subreddit_user_list_test(test_df,
                                          'requester_subreddits_at_request',
                                          subreddit_freq)
subreddit_fill_zeroes(subreddit_user_list, test_df.shape[0])
test_subreddit_user_list = pd.DataFrame(subreddit_user_list,
                                        index = range(test_df.shape[0]))

# Perform PCA on the "subreddit" features only. Selected 260 as the number of
# components based on experimentation, which revealed that about 70% of variance
# is explained by these many components.
# print(test_subreddit_user_list)
pca = PCA(n_components = 260)
train_data_pca = pca.fit_transform(train_subreddit_user_list)
# train_data_pca = pd.DataFrame(train_data_pca,columns=list())
test_data_pca = pca.transform(test_subreddit_user_list)
# test_data_pca = pd.DataFrame(test_data_pca,columns=['subreddit_proba'])
print(train_data_pca)
# Train a logistic regression model.
logistic = LogisticRegression()
logistic.fit(train_data_pca, train_labels)
#
# # Predict the probabilities for the training and test data sets for both the classes,
# # and concatenate these probabilities as a new structured feature into the original
# # dataframe consisting of other structured features and corresponding dummy variables
# # for the categorical data.
predictions = logistic.predict_proba(train_data_pca)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

train_data_subreddit = pd.concat([train_data,
                                 pd.DataFrame({'subreddit_proba': predictions[:, 0]})],axis=1)
predictions = logistic.predict_proba(test_data_pca)

test_data_subreddit = pd.concat([test_data,
                                 pd.DataFrame({'subreddit_proba': predictions[:, 0]})],axis=1)

# print(train_data_subreddit.columns.values,len(train_data_subreddit.columns.values))

# train_data_subreddit = pd.concat([train_data, \
#                                   train_subreddit_user_list],axis=1)
# test_data_subreddit = pd.concat([test_data, \
#                                  test_subreddit_user_list],axis=1)

count_vect = CountVectorizer(analyzer=preprocess_text)
# Extract text features from the post text by generating features from words
train_count_vectors = count_vect.fit_transform(
                          df_train['request_text_edit_aware'].values)
test_count_vectors = count_vect.transform(
                         df_test['request_text_edit_aware'].values)
# # print(count_vect.vocabulary_)
#
# # Turn the above sparse matrix representation into dense, as PCA expects
train_count_vectors = train_count_vectors.toarray()
test_count_vectors = test_count_vectors.toarray()

# Performed experimenation with PCA and settled on 500 features as they captured
# 80% of the variance of the text features.
pca = PCA(n_components = 500)
train_count_vectors_pca = pca.fit_transform(train_count_vectors)
# train_count_vectors_pca = pd.DataFrame(train_count_vectors_pca,columns=['text_proba'])
test_count_vectors_pca = pca.transform(test_count_vectors)
# test_count_vectors_pca = pd.DataFrame(test_count_vectors_pca,columns=['text_proba'])

# Train a logistic regression model on just the text features from post text.
logistic = LogisticRegression()
logistic.fit(train_count_vectors_pca, train_labels)
#
# # Predict the probabilities for the training and test data sets for both the classes,
# # and concatenate these probabilities as a new structured feature into the original
# # dataframe consisting of other structured features and corresponding dummy variables
# # for the categorical data.
predictions = logistic.predict_proba(train_count_vectors_pca)
train_data_subreddit_count = pd.concat([train_data_subreddit,
                                 pd.DataFrame({'text_proba': predictions[:, 0]})],axis=1)
predictions = logistic.predict_proba(test_count_vectors_pca)
test_data_subreddit_count = pd.concat([test_data_subreddit,
                                 pd.DataFrame({'text_proba': predictions[:, 0]})],axis=1)

# train_data_subreddit_count = pd.concat([train_data_subreddit,
#                                         train_count_vectors],axis=1)
# test_data_subreddit_count = pd.concat([test_data_subreddit,
#                                        test_count_vectors],axis=1)


# Train the final logistic regression model on the full set of features
logistic = LogisticRegression()
# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier(loss='hinge', penalty='l1',
#                                             alpha=0.01, random_state=42,
#                                               max_iter=50, tol=None)
# pd_target = pd.DataFrame(train_labels,columns=['requester_received_pizza'])
# train_data_subreddit_count = pd.concat([train_data_subreddit_count,\
#                                  pd_target],axis=1)

train_data_scaled = preprocessing.scale(train_data_subreddit_count.astype(float))
test_data_scaled = preprocessing.scale(test_data_subreddit_count.astype(float))

train_data_subreddit_count.to_csv("traindata.csv")
logistic.fit(train_data_scaled, train_labels)

predictions = logistic.predict(test_data_scaled)

# Turn predictions into a simple integer list for easy printing into a submission format
predictions = [int(prediction) for prediction in list(predictions)]

# Extract the request_id to incorporate into the submission file
request_id = list(test_df['request_id'].values)

# Create submission matrix
data = [['request_id', 'requester_received_pizza']]
for i in range(len(predictions)):
    data.append([str(request_id[i]),str(predictions[i])])

# Output to csv
with open('final.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)

logistic = LogisticRegression()
logistic.fit(train_data_scaled, train_labels)
# print(train_data_subreddit_count.columns.values)

coefs = pd.DataFrame(list(zip(train_data_subreddit_count.columns.values, np.transpose(logistic.coef_))))
coefs.columns = ['Variable', 'Coefficient']
coefs = coefs.sort_values(by=['Coefficient'])

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.max_rows')

print_full(coefs)

# from sklearn.model_selection import train_test_split
# train, valid = train_test_split(train_data_subreddit_count, test_size=0.2,random_state=42)
# # from sklearn.naive_bayes import MultinomialNB
# # clf = MultinomialNB()
# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier(loss='hinge', penalty='l1',
#                                             alpha=0.01, random_state=42,
#                                               max_iter=50, tol=None)
# # from sklearn.neural_network import MLPClassifier
# # clf=MLPClassifier()
# # # # To train the classifier, simple do
# train_tags = train['requester_received_pizza']
# valid_tags = valid['requester_received_pizza']
#
# train = preprocessing.scale(train)
# valid = preprocessing.scale(valid)
# clf.fit(train, train_tags)
# from sklearn.metrics import accuracy_score
# # #
# # # # To predict our tags (i.e. whether requesters get their pizza),
# # # # we feed the vectorized `test_set` to .predict()
# predictions_valid = clf.predict(valid)
# # print(len(predictions_valid))
# # print(sum(predictions_valid))
# #
# print('Pizza reception accuracy = {}'.format(
#          accuracy_score(predictions_valid, valid_tags) * 100)
#       )
