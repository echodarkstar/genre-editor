from flask import Flask
from flask import render_template, flash, redirect, request, url_for, jsonify
from nltk.corpus import wordnet

app = Flask(__name__)
from app import app
import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
import time

from app.pplm_classification_head import ClassificationHead
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel
from app.run_pplm import full_text_generation
from app.run_pplm import run_pplm_example
from app.run_pplm import get_classifier
from app.run_pplm import get_bag_of_words_indices
from app.run_pplm import generate_text_pplm
import logging
import copy

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('app.log', 'a'))

logger.info("#"*15 + "RUN STARTED" + "#"*15)
# set Random seed
torch.manual_seed(42)
np.random.seed(42)
# set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(device)

# load pretrained model
pretrained_model="gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
model.to(device)
model.eval()

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False

logger.info("Model loaded!")

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'MindBender'}
    return render_template('index.html', title='Home', user=user)

@app.route('/generate', methods=['POST'])
def generate():
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    start = time.time()
    torch.cuda.empty_cache()
    request.json['text'] = request.json['text'].strip()
    logger.info(request.json['genre'])
    logger.info(request.json['text'])
    tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + request.json['text'])
    bag_of_words = (request.json['genre']).lower()

    params = {k:num(v) for k,v in json.loads(request.json['params']).items()}
    logger.info(params)
    #Arguments
    context=tokenized_cond_text
    num_samples=1
    bag_of_words= bag_of_words
    discrim=None
    class_label=-1
    length= params['length']
    stepsize= params['stepsize']
    temperature= params['tempSlider']
    top_k= params['topk']
    sample=True
    num_iterations= params['iterations']
    grad_length= params['gradl']
    horizon_length= params['horil']
    window_length= params['windl']
    decay=False
    gamma= params['gamma']
    gm_scale= params['gms']
    kl_scale= params['kls']
    repetition_penalty= params['rep']
    
    classifier, class_id = get_classifier(discrim, class_label, device)
    bow_indices = []
    PPLM_BOW = 1
    PPLM_DISCRIM = 2
    PPLM_BOW_DISCRIM = 3

    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"), tokenizer)

    if bag_of_words and classifier:
        # logger.info("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif bag_of_words:
        loss_type = PPLM_BOW
        # logger.info("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        # logger.info("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            repetition_penalty=repetition_penalty,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()
    # untokenize unperturbed text
    pert_gen_text = [tokenizer.decode(x.tolist()[0]) for x in pert_gen_tok_texts]

    generated_texts = []
    colorama=False
    bow_word_ids = set()
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"), tokenizer)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ""
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += "{}{}{}".format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL,
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id])
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

            # logger.info("= Perturbed generated text {} =".format(i + 1))
            # logger.info(pert_gen_text)
        except Exception as exc:
            pass
            # logger.info("Ignoring error while generating perturbed text:", exc)

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append([tokenized_cond_text, pert_gen_tok_text])

    # run_pplm_example(
    #     cond_text=request.json['text'],
    #     num_samples=1,
    #     bag_of_words=bag_of_words,
    #     length=50,
    #     stepsize=0.03,wor
    #     sample=False,
    #     num_iterations=3,
    #     window_length=5,
    #     gamma=1.5,
    #     gm_scale=0.95,
    #     kl_scale=0.01,
    #     seed=42
    # )
    logger.info("Total time {}".format(time.time() - start))
    prompt = [x[0] for x in generated_texts][0]
    preds = [x[1][0] for x in generated_texts]
    logger.info("-------------Prompt---------------")
    logger.info(tokenizer.decode(prompt).replace('<|endoftext|>', ''))
    logger.info("-------------Prediction-----------------")
    pred_text = [tokenizer.decode(x[len(prompt):]).replace('<|endoftext|>', '') for x in preds]
    for pred in pred_text:
        logger.info(pred)
    logger.info("-------------Total-----------------")
    total_text = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in preds]
    for tot in total_text:
        logger.info(tot)
    return jsonify({'success':True, 'text':pred_text[0]}), 200, {'ContentType':'application/json'} 

@app.route('/synonyms', methods=['POST'])
def synonyms():
    synonyms = []
    request.json['selectedText'] = request.json['selectedText'].strip()
    if request.json['selectedText']:
        for syn in wordnet.synsets(request.json['selectedText']): 
            for l in syn.lemmas(): 
                if (l.name().lower() != request.json['selectedText'].lower()) and l.name() not in synonyms:
                    synonyms.append(l.name()) 
        return jsonify({'success':True, 'synonyms':synonyms}), 200, {'ContentType':'application/json'}
    print(request.json['selectedText'])
    return jsonify({'success':True, 'synonyms':"", 'synonyms1':""}), 200, {'ContentType':'application/json'} 

