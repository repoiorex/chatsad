import nltk 
from flask import Flask, render_template, request
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.ops import array_ops 
stemmer = LancasterStemmer()
import tensorflow as tf
from datetime import datetime 
import numpy
import tflearn
import tensorflow
import json
import random
import pickle 
nltk.download('punkt')
import time
import threading
import sched

from flask import Flask, render_template, request, jsonify
import os,sys,requests, json
from random import randint

with open("contenido.json", encoding='utf-8') as archivo:#abrir el archivo contenido.json
    datos = json.load(archivo)

#print(datos)#imprime los datos de contenido.json

    


palabras=[]
tags=[]
auxX=[]  
auxY=[]

for contenido in datos["contenido"]:
    for patrones in contenido["patrones"]:
        auxpalabra = nltk.word_tokenize(patrones) #almacela la palabra
        palabras.extend(auxpalabra)
        auxX.append(auxpalabra)
        auxY.append(contenido["tag"])

        if contenido["tag"] not in tags:
            tags.append(contenido["tag"])

palabras = [stemmer.stem(w.lower()) for w in palabras if w!="?"]
palabras = sorted(list(set(palabras)))
tags = sorted(tags) 
    #print(tags)

entrenamiento = []
salida = []
salidaVacia=[0 for _ in range(len(tags))]

for x, documento in enumerate(auxX):
    cubeta=[]
    auxpalabra=[stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxpalabra:
            cubeta.append(1)
        else:
            cubeta.append(0)
            filaSalida = salidaVacia[:]    
            filaSalida[tags.index(auxY[x])] =1 
            entrenamiento.append(cubeta)
            salida.append(filaSalida)

entrenamiento = numpy.array(entrenamiento) 
salida = numpy.array(salida)


    #print(entrenamiento)
    #print(salida)
tf.compat.v1.reset_default_graph()


red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red,20)
red = tflearn.fully_connected(red,20)
red = tflearn.fully_connected(red,len(salida[0]), activation="softmax") 
red = tflearn.regression(red)

modelo = tflearn.DNN(red)


modelo.fit(entrenamiento,salida, n_epoch=1000, batch_size=20, show_metric=True)
modelo.save("modelo.tflearn")   

interaccion = ["mensaje 1","mensaje 2","mensaje 3","mensaje 4","mensaje 5"]
contadior = ""
repetir=[]
prueba=[]
entrada = None

#def Respuesta (entrada):
    
    #entrada = input('msg')

def Espera(entrada):
   
    for i in range(10):
        if entrada == None:
            time.sleep(1)
        else:
            pass


def ValidarEntrada(entrada):
    ofertas = random.choice(interaccion)
    
    Espera(entrada)
    if entrada == None:
        print("\n", ofertas)


def mainBot(entrada):
    #global entrada
    contag = ""
    cont = 0
    p = False
    print("Hola, en que te puedo apoyar?")
    #print(entrada)
    #while True:
        #entrada = None
        #
        # start_time = datetime.now()  
        #h1 = threading.Thread(target=Respuesta)
        #h1.start()
        #while entrada == None:
            #h2 = threading.Thread(target=ValidarEntrada(entrada))
            #/h2.start()
            #h2.join()

        #h1.join()
                
            
    cubeta = [0 for _ in range(len(palabras))]
    entradaProcesada = nltk.word_tokenize(entrada)
    entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
    for palabraIndividual in entradaProcesada:
        for i,palabra in enumerate(palabras):
            if palabra == palabraIndividual:
                cubeta[i] = 1
    resultados = modelo.predict([numpy.array(cubeta)])
    resultadosIndices = numpy.argmax(resultados)            
    tag = tags[resultadosIndices]

    for i in range(len(resultados[0])):
        if resultados[0][i] > 0.7 :
            p = True

            if p == True:
                for tagAux in datos["contenido"]:
                    if tagAux["tag"] == tag:
                        if tagAux["tag"] != contag:
                            contag= tagAux["tag"]
                            respuesta = tagAux["respuestas"]
                            return random.choice(respuesta)
                        else:  
                            return("ya dijisti eso")  
        else:
           return("Parece ser que lo que escribiste no lo entiendo, te puede apoyar con ")  

app = Flask(__name__)
app.static_folder='static'

@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/get")
def mainBotresponse():
	userText = request.args.get('msg')
	return mainBot(userText)

if __name__ == "__main__":
    app.run()
mainBot()



