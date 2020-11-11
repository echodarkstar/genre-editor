from flask import Flask
from flask import render_template, flash, redirect, request, url_for, jsonify

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
