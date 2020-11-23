from flask import Flask
from flask import render_template, flash, redirect, request, url_for, jsonify
from nltk.corpus import wordnet

from app import app

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Rishabh'}
    return render_template('index.html', title='Home', user=user)

@app.route('/generate', methods=['POST'])
def generate():
    print(request.json['genre'])
    print(request.json['text'])
    return jsonify({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/synonyms', methods=['POST'])
def synonyms():
    synonyms = []
    if request.json['selectedText']:
        for syn in wordnet.synsets(request.json['selectedText']): 
            for l in syn.lemmas(): 
                synonyms.append(l.name()) 
        return jsonify({'success':True, 'synonyms':synonyms[0], 'synonyms1':synonyms[1]}), 200, {'ContentType':'application/json'}
    print(set(synonyms)) 
    print(request.json['selectedText'])
    return jsonify({'success':True, 'synonyms':"", 'synonyms1':""}), 200, {'ContentType':'application/json'} 

