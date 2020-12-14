# genre-editor
A text editor with generative capabilities that allows the user to generate text in different genres.

## Table of contents
* [About the Project](#about-the-project)
* [System Design](#system-design)
* [Technologies](#technologies)
* [Setup](#setup)

## About the Project
Recent advances in natural language processing have been adopted widely in various fields and products. Natural language generation in particular has been used to display real-time predictive text while typing on QWERTY keypads, search engines, and even email. In this project, we utilize recent progress on topic specific language generation to develop a proof of concept implementation of a word processor with predictive capabilities. We also present a heuristic evaluation of our system, and propose modifications based on user feedback we gathered.

Our application is built using HTML, CSS/Bootstrap, and Javascript. The Javascript is used to make the editor dynamic and communicate with Flask endpoints. Our particular appliation runs on Flask, but the design is such that the Flask portion can be rewritten as REST APIs that allow the front-end to be possibly decoupled from the back-end. 
Bootstrap is a framework that helps in faster design and prototyping of web application front-ends. Bootstrap's responsive CSS adjusts to phones, tablets, and desktops and hence users will be benefited by utilizing it on any device.Users can use the Sticky Notes feature to scribble notes or jot down ideas before carrying it over to the finished draft. 
Since the user might want to save the contents, a Save button is provided which saves the file in Docx format retaining all formatting. An upload file button is also provided. The Word Counter allows the user to keep a check on how much they've written. For writers who prefer to use keyboards rather than a mouse, we have included the common keyboard shortcuts for text styling. 

UberAI developed an alternative to extensive fine-tuning: simple attribute models (classifiers) based on user-defined wordlists. They refer to this model as [PPLM (Plug and Play Language Models)](https://github.com/uber-research/PPLM). The sampling process involves a forward and backward pass where gradients from this attribute model are used to change the hidden representations in the language model. In this case, the language model used is GPT-2. The following diagram is reproduced from their [paper](https://arxiv.org/abs/1912.02164).

[![pplm-exp.png](https://i.postimg.cc/CxC1KvLN/pplm-exp.png)](https://postimg.cc/CR1YPs8B)

Let us go over the two Flask endpoints: /synonyms and /generate.
* /synonyms: We use nltk to retrieve the synonyms of the selected word. A POST request to this endpoint is sent everytime the user selects a word.
* /generate: Two AJAX POST requests are sent to this endpoint when the user selects text that contains atleast two words and presses the **Alt** key. The POST payload consists of the currently selected genre and the selected text. Inside the endpoint, the model takes this as input and proceeds to perform the updates as described in the previous section.

## System Design

[![dig.png](https://i.postimg.cc/QxkXc0vn/dig.png)](https://postimg.cc/vcTFsLYW)
	
## Technologies

Project is created with:
* Flask
* Bootstrap
* NLTK
* Pytorch
	
## Setup

Install prerequisites:

```
pip install -r requirements.txt
```

Run the following commands from the project directory to get started:

```
source setup.sh
flask run
```