import logging
import numpy as np
import sys
import warnings
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import mean_squared_error
app=Flask(__name__)
@app.route("/")
def homepage():
	return render_template("index.html")
	
	
@app.route("/services.html")
def final():
	return render_template("services.html")
	
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[0, :].nonzero()[0], size=4, replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    assert(np.all((train*test) == 0))
    return train, test
	
def similarity(ratings, kind='user'):
    if kind == 'user':
        sim = ratings.dot(ratings.T)
    elif kind == 'issue':
        sim = ratings.T.dot(ratings)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return sim/norms/norms.T

def top_k_issueNames(similarity, mapper, issueName_idx, k=3):
    return [mapper[x] for x in np.argsort(similarity[issueName_idx,:])[:-k-1:-1]]

@app.route("/",methods=['POST'])
def getvalue():
	uid = request.form['uid']
	names = ['user_id', 'issue_id', 'issues']
	df = pd.read_csv('u.data', sep='\t', names=names)
	n_users = df.user_id.unique().shape[0]
	n_issues = df.issue_id.unique().shape[0]
	ratings = np.zeros((n_users, n_issues))

	for row in df.itertuples():
		ratings[row[1]-1, row[2]-1] = row[3]
	train, test = train_test_split(ratings)
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	user_similarity = similarity(train, kind='user')
	issue_similarity = similarity(train, kind='issue')
	
	idx_to_issueName = {}
	with open('u.item', 'r', encoding='ISO-8859-1') as f:
		for line in f.readlines():
			info = line.split('|')
			idx_to_issueName[int(info[0])] = info[1]
    
	idx = int(uid)
	issueNames = top_k_issueNames(issue_similarity, idx_to_issueName, idx-1)
	aa=issueNames[0]
	bb=issueNames[1]
	cc=issueNames[2]
	
	return render_template("result.html", a=aa, b=bb, c=cc)



if __name__ == '__main__':
	app.run(host='127.0.0.1' , port=8080 , debug=True)