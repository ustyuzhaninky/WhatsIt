# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:13:01 2018

@author: Ustyuzhanin K. Yu
"""

import os
import itertools
import numpy as np
import pandas as pd
import h5py

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import model_from_yaml
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import plot_model
from keras.utils import multi_gpu_model

import tensorflow as tf

from . import filetools as ft

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          spath='confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(spath)

class predictor(object):

    MODEL_STRUCTURE_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure.yaml')
    MODEL_WEIGHTS_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weight.HDF5')
    model = None
    cnf_matrix = None
    class_names = None
    train_datagen = None
    
    def __init__(self):
        self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        self.model = Sequential()
    
    def save(self): 
            
        model_filepath = self.MODEL_STRUCTURE_NAME
        weights_filepath = self.MODEL_WEIGHTS_NAME
        yaml_string = self.model.to_yaml()
        with open(model_filepath,'w') as stream:
            try:
                stream.write(yaml_string)
                stream.close()
            except yaml.YAMLError as exc:
                print('File can not be readed...')
            self.model.save_weights(weights_filepath)

    def load(self):

        model_filepath = self.MODEL_STRUCTURE_NAME
        weights_filepath = self.MODEL_WEIGHTS_NAME
        with open(model_filepath,'r') as stream:
            try:
                yaml_string = stream.read()
            except yaml.YAMLError as exc:
                print('File can not be readed...')
            self.model = model_from_yaml(yaml_string)
            self.model.load_weights(weights_filepath, by_name=False)
        
    def predict(self, data):
#        return keras.applications.inception_v3.decode_predictions(
#                self.model.predict(data))
        return self.model.predict(data)
    
    def create(self, epoches=1):
        traindata = 'trains'
        testdata = 'tests'
        print('Creation begins...')
        print('stage 1')
        #with tf.device('/cpu:0'):
        self.inseption = keras.applications.inception_v3.InceptionV3(include_top=False, 
                                                        weights='imagenet', 
                                                        #input_tensor=None,
                                                        input_shape=((150, 150, 3)),
                                                        #pooling=None,
                                                        classes=5)
        
        self.inseption.trainable = False
        print(self.inseption.summary())
        #input('Enter anything to start training')
#        self.bottleneck_datagen = ImageDataGenerator(rescale=1./255)
#        print('stage 2')
#        self.train_generator = self.train_datagen.flow_from_directory(
#                traindata,
#                target_size=(150, 150),
#                batch_size=32,
#                class_mode = None,
#                shuffle=False
#                )
#        print('stage 3')
#        self.validation_generator = self.bottleneck_datagen.flow_from_directory(testdata,
#                                                               target_size=(150, 150),
#                                                               batch_size=32,
#                                                               class_mode=None,
#                                                               shuffle=False)
#        print('stage 4')
#        if not os.path.exists('bottleneck_features/bn_features_train.npy'):
#            print('Creating bottleneck_features/bn_features_train.npy')
#            bottleneck_features_train = self.inseption.predict_generator(self.train_generator, 501)
#            ft.crfile('bottleneck_features/bn_features_train.npy')
#            np.save(open('bottleneck_features/bn_features_train.npy', 'wb+'), bottleneck_features_train)
#        else:
#            print('File bottleneck_features/bn_features_train.npy already here.')
#        
#        if not os.path.exists('bottleneck_features/bn_features_validation.npy'):
#            print('Creating bottleneck_features/bn_features_validation.npy')
#            bottleneck_features_validation = self.inseption.predict_generator(self.validation_generator, 501)
#            ft.crfile('bottleneck_features/bn_features_validation.npy')
#            np.save(open('bottleneck_features/bn_features_validation.npy', 'wb+'), bottleneck_features_validation)
#        else:
#            print('File bottleneck_features/bn_features_validation.npy already here.')
#        
#        if os.path.exists('bottleneck_features/bn_features_train.npy'):
#            print('Loading bottleneck_features/bn_features_train.npy')
#            train_data = np.load(open('bottleneck_features/bn_features_train.npy', 'rb+'))
#        else:
#            print('File bottleneck_features/bn_features_train.npy does not exist.')
#        
#        print(train_data.shape)
#        self.train_labels = np.array([0] * 15629 + [1] * 15629 + [2] * 15629 + [3] * 15629 + [4] * 15629) 
#        self.train_labels = np.reshape(self.train_labels, (15629, 5))
#        
#        #self.train_labels = np.array([[0] * 50, [1] * 50,[2] * 50, [3] * 50, [4] * 50]) 
#        #self.train_labels = np.reshape(self.train_labels, (self.train_labels.shape[1], self.train_labels.shape[0]))
#        
#        if os.path.exists('bottleneck_features/bn_features_validation.npy'):
#            print('Loading bottleneck_features/bn_features_validation.npy')
#            validation_data = np.load(open('bottleneck_features/bn_features_validation.npy', 'rb+'))
#        else:
#            print('File bottleneck_features/bn_features_validation.npy does not exist.')
#        
#        
#        self.validation_labels = np.array([0] * 15539 + [1] * 15539 + [2] * 15539 + [3] * 15539 + [4] * 15539) 
#        self.validation_labels = np.reshape(self.validation_labels, (15539, 5))
#        
#        #self.validation_labels = np.array([[0] * 50, [1] * 50, [2] * 50, [3] * 50, [4] * 50]) 
#        #self.validation_labels = np.reshape(self.validation_labels, (self.validation_labels.shape[1], self.validation_labels.shape[0]))
#        
#        print('stage 5')
#        self.model = Sequential()
#        self.model.add(Flatten(input_shape=train_data.shape[1:]))
#        self.model.add(Dense(64, activation='relu', name='dense_one'))#256 neurons in original
#        self.model.add(Dropout(0.5, name='dropout_one'))
#        self.model.add(Dense(64, activation='relu', name='dense_two'))#256 neurons in original
#        self.model.add(Dropout(0.5, name='dropout_two'))
#        self.model.add(Dense(5, activation='softmax', name='output'))
#        print('stage 6')
#        self.model.compile(optimizer='rmsprop', 
#                      loss='categorical_crossentropy', 
#                      metrics=['accuracy'])
#        print('stage 7')
#        self.model.fit(train_data, self.train_labels,
#            epochs=50, batch_size=32,
#            validation_data=(validation_data, self.validation_labels))
#        
#        self.save()
        print('stage 8')
        #with tf.device('/cpu:0'):
        self.model = Sequential()
        self.model.add(self.inseption)
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', name='dense_one'))
        self.model.add(Dropout(0.5, name='dropout_one'))
        self.model.add(Dense(64, activation='relu', name='dense_two'))
        self.model.add(Dropout(0.5, name='dropout_two'))
        self.model.add(Dense(5, activation='softmax', name='output'))
        #self.model = multi_gpu_model(self.model, gpus=1)
#        x = Flatten()(self.inseption.output)
#        x = Dense(64, activation='relu', name='dense_one')(x)
#        x = Dropout(0.5, name='dropout_one')(x)
#        x = Dense(64, activation='relu', name='dense_two')(x)
#        x = Dropout(0.5, name='dropout_two')(x)
#        top_model=Dense(5, activation='sigmoid', name='output')(x)
#        self.model = Model(input=self.inseption.input, output=top_model)
        #self.model = model_from_yaml(self.MODEL_STRUCTURE_NAME)
        
        for layer in self.inseption.layers[:205]:
            layer.trainable = False
        
        #self.model.load_weights(self.MODEL_WEIGHTS_NAME, by_name=True)
        self.model.compile(loss='categorical_crossentropy',
              #optimizer=Adam(lr=1e-5),
              #optimizer=SGD(lr=1e-1, momentum=0.9),
              optimizer=RMSprop(lr=2e-1),
              metrics=['accuracy'])
        
        print(self.model.summary())
        print('stage 9')
        filepath="new_model_weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max')
        callbacks_list = []#checkpoint]
        print('stage 10')
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        print('stage 11')
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = self.train_datagen.flow_from_directory(
                traindata,
                target_size=(150, 150),
                batch_size=1,
                class_mode='categorical')
        
        self.validation_generator = self.test_datagen.flow_from_directory(
                testdata,
                target_size=(150, 150),
                batch_size=1,
                class_mode='categorical')
        
        
        self.pred_generator = self.test_datagen.flow_from_directory('tests/',
                                                             target_size=(150,150),
                                                             batch_size=1,
                                                             class_mode='categorical')
        print('stage 12')
        self.model.fit_generator(
                self.train_generator,
                samples_per_epoch=500,
                nb_epoch=epoches,#100,
                validation_data=self.validation_generator,
                nb_val_samples=32,
                callbacks=callbacks_list)
        print('Creation ended')
    
    def train(self, epoches = 100):
        traindata = 'trains'
        testdata = 'tests'
        
        self.model = Sequential()
        mm.load_weights(self.MODEL_WEIGHTS_NAME, by_name=False)
        self.model.add(mm)
        for layer in mm.layers[:205]:
            layer.trainable = False
        self.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-5), 
              metrics=['acc'])
        
        self.train_generator = self.train_datagen.flow_from_directory(
                traindata,
                target_size=(150, 150),
                batch_size=32,
                class_mode='categorical')
        
        self.validation_generator = self.test_datagen.flow_from_directory(
                testdata,
                target_size=(150, 150),
                batch_size=32,
                class_mode='categorical')
        self.model.fit_generator(
                self.train_generator,
                samples_per_epoch=500,
                nb_epoch=epoches,
                validation_data=self.validation_generator,
                nb_val_samples=500)
    
    def testme(self, ytest, ypred):
        self.cnf_matrix = pd.crosstab(ytest, ypred)
        self.class_names = [0, 1, 2, 3, 4]
        print('Classes counts:')
        print(pd.value_counts(ypred))
        self.cnf_matrix = confusion_matrix(ytest, ypred)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(self.cnf_matrix, classes=self.class_names,
                      title='Confusion matrix, without normalization', spath='confusion.png')
        plot_confusion_matrix(self.cnf_matrix, classes=self.class_names, normalize=True,
                      title='Normalized confusion matrix', spath='confusion_normed.png')
        #self.evaluation_test = self.model.evaluate_generator(self.pred_generator, val_samples=100)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        self.pred_generator = self.test_datagen.flow_from_directory('Mix/',
                                                             target_size=(150,150),
                                                             batch_size=100,
                                                             class_mode='categorical')
        
        self.imgs,  self.labels=self.pred_generator.next()
        self.array_imgs=np.transpose(np.asarray([img_to_array(img) for img in self.imgs]),(0,2,1,3))
        self.predictions=self.model.predict(self.imgs)
        #self.rounded_pred=np.asarray([round(i) for i in self.predictions])
    
    def plotme(self):
        plot_model(self.model, show_shapes=True, show_layer_names=True, to_file='results/model.png')
    
    def summary(self):
        keras.utils.print_summary(self.model, line_length=None, positions=None, print_fn=None)
    
    def showwrong(self):
        wrong=[im for im in zip(self.imgs, self.labels, self.predictions) if im[1].all()!=im[2].all()]
        plt.figure()
        plt.figure(figsize=(12,12))
        for ind, val in enumerate(wrong[:25]):
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0.2, hspace = 0.2)
            plt.subplot(5,5,ind+1)
            im=val[0]
            plt.axis('off')
            plt.text(0, 0, val[1], fontsize=11, color='red')
            ft.crfile('results/right20/right_image{0}.png'.format(ind))
            plt.savefig('results/right20/right_image{0}.png'.format(ind))
            #plt.text(0, 155, val[1], fontsize=11, color='red')
            #ft.crfile('results/r/mistake_image{0}.png'.format(ind))
            #plt.savefig('results/mistakes/mistake_image{0}.png'.format(ind))
            plt.imshow(im)
            
    def showright(self):
        right=[im for im in zip(self.imgs, self.labels, self.predictions) if im[1].all()==im[2].all()]
        plt.figure()
        plt.figure(figsize=(12,12))
        for ind, val in enumerate(right[:20]):
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0.2, hspace = 0.2)
            plt.subplot(5,5,ind+1)
            im=val[0]
            plt.axis('off')
            plt.text(0, 0, val[1], fontsize=11, color='red')
            #plt.text(0, 155, val[2], fontsize=11, color='red')
            ft.crfile('results/right20/right_image{0}.png'.format(ind))
            plt.savefig('results/right20/right_image{0}.png'.format(ind))
            plt.imshow(im)

def imgPreload(img):
    preloaded = image.load_img(img, target_size=(150, 150))
    preloaded = image.img_to_array(preloaded)
    #preloaded = np.expand_dims(preloaded, axis=0)
    print(preloaded.shape)
    return preloaded

def main(*args):
    testdata = 'tests'
    testclasses = []   
    with open(os.path.join(testdata,'index.txt'), 'r') as classes_string:
        testclasses.append(classes_string.readline())
    files = ft.extract_files(testdata)
    
    test_images = []
    for file in files:
        preloaded = image.load_img(file, target_size = (150,150))
        preloaded = image.img_to_array(preloaded)
        preloaded = np.expand_dims(preloaded, axis = 0)
        test_images.append(preloaded)
        
    prd = predictor()
    prd.create()
    prd.save()
    
    prd.load()
    print(prd.model.summary())
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory('Mix',
                                            target_size=(150,150),
                                            batch_size=100,
                                            class_mode='categorical')
    imgs, labels = generator.next()
    #imgs = np.transpose(np.asarray([img_to_array(img) for img in imgs]),(0,2,1,3))
    res = prd.predict(imgs)
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels[i][j] *= j
    
    res = np.reshape(res, (res.shape[0]*res.shape[1]))
    labels = np.reshape(labels, (labels.shape[0]*labels.shape[1]))
    print('res:',res)
    print('lables', labels)
    res=np.asarray([round(i) for i in res])
    
    prd.testme(labels, res)
    prd.showwrong()
    prd.showright()

#if __name__ == '__main__':
#    #print(keras.applications.imagenet_utils.CLASS_INDEX_PATH)
#    main()