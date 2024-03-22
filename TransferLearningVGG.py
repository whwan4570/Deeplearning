import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import tensorflow as tf
from keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from keras_preprocessing.image import ImageDataGenerator, img_to_array

TRAIN_DIR = 'data/images/images'

artists_df = pd.read_csv('data/artists.csv')
artists_df.drop(['id', 'bio', 'wikipedia'], axis=1, inplace=True)

artists_df = artists_df.sort_values(by=['paintings'], ascending=False)
artists_top = artists_df[artists_df['paintings'] >= 200].reset_index()
artists_top = artists_top[['name', 'paintings']]
artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)

class_weights = artists_top['class_weight'].to_dict()

updated_name = "Albrecht_Dürer".replace("_", " ")
artists_top.iloc[4, 0] = updated_name

artists_top_name = artists_top['name'].str.replace(' ', '_')

# Print few random paintings
n = 5
fig, axes = plt.subplots(1, n, figsize=(10, 30), dpi=300)

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(TRAIN_DIR, random_artist)))
    random_image_file = os.path.join(TRAIN_DIR, random_artist, random_image)
    image = plt.imread(random_image_file)
    axes[i].imshow(image)
    axes[i].set_title("Artist: " + random_artist.replace('_', ' '))
    axes[i].axis('off')

plt.show()

batch_size = 32
train_input_shape = (224, 224, 3)
n_classes = artists_top.shape[0]

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1. / 255.,
                                   shear_range=5,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   )

train_generator = train_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                    )

valid_generator = train_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                    )

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
print(f"Total number of Train Batches: {STEP_SIZE_TRAIN}, Validation Batches: {STEP_SIZE_VALID}")

# Print a random paintings and it's random augmented version
fig, axes = plt.subplots(1, 2, figsize=(20,10))

random_artist = random.choice(artists_top_name)
random_image = random.choice(os.listdir(os.path.join(TRAIN_DIR, random_artist)))
random_image_file = os.path.join(TRAIN_DIR, random_artist, random_image)

# Original image
image = plt.imread(random_image_file)
axes[0].imshow(image)
axes[0].set_title("An original Image of " + random_artist.replace('_', ' '))
axes[0].axis('off')

# Transformed image
aug_image = train_datagen.random_transform(image)
axes[1].imshow(aug_image)
axes[1].set_title("A transformed Image of " + random_artist.replace('_', ' '))
axes[1].axis('off')

plt.show()

resnet_base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

for layer in resnet_base_model.layers:
    layer.trainable = True

# Add layers at the end
X_resnet = resnet_base_model.output
X_resnet = Flatten()(X_resnet)

X_resnet = Dense(512, kernel_initializer='he_uniform')(X_resnet)
# X = Dropout(0.5)(X)
X_resnet = BatchNormalization()(X_resnet)
X_resnet = Activation('relu')(X_resnet)

X_resnet = Dense(16, kernel_initializer='he_uniform')(X_resnet)
# X = Dropout(0.5)(X)
X_resnet = BatchNormalization()(X_resnet)
X_resnet = Activation('relu')(X_resnet)

resnet_output = Dense(n_classes, activation='softmax')(X_resnet)

resnet_model = Model(inputs=resnet_base_model.input, outputs=resnet_output)

optimizer = Adam(lr=0.0001)
resnet_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

n_epoch = 10
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           mode='auto', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='auto')

resnet_history1 = resnet_model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr],
                              class_weight=class_weights
                             )

# Freeze core ResNet layers and train again
for layer in resnet_model.layers:
    layer.trainable = False

for layer in resnet_model.layers[:50]:
    layer.trainable = True

optimizer = Adam(lr=0.0001)

resnet_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 10
resnet_history2 = resnet_model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr, early_stop],
                              class_weight=class_weights
                             )

# Merge history1 and history2
resnet_history = {}
resnet_history['loss'] = resnet_history1.history['loss'] + resnet_history2.history['loss']
resnet_history['acc'] = resnet_history1.history['accuracy'] + resnet_history2.history['accuracy']
resnet_history['val_loss'] = resnet_history1.history['val_loss'] + resnet_history2.history['val_loss']
resnet_history['val_acc'] = resnet_history1.history['val_accuracy'] + resnet_history2.history['val_accuracy']
resnet_history['lr'] = resnet_history1.history['lr'] + resnet_history2.history['lr']

plt.style.use('seaborn')


# Plot the training graph
def plot_training(history, model_name):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')
    fig.suptitle(f'Model: {model_name}')
    plt.savefig(f'{model_name}_plots.png')
    plt.show()


plot_training(resnet_history, 'ResNet50')


# Prediction accuracy on CV data
resnet_score = resnet_model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV Data: ", resnet_score[1])


# get predictions
def get_predictions_and_labels(model, valid_generator, STEP_SIZE_VALID):
    # Loop on each generator batch and predict
    y_pred, y_true = [], []
    for i in range(STEP_SIZE_VALID):
        (X, y) = next(valid_generator)
        y_pred.append(model.predict(X))
        y_true.append(y)

    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]

    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()

    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()
    return y_pred, y_true


p_valid, y_valid = get_predictions_and_labels(resnet_model, valid_generator, STEP_SIZE_VALID)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(p_valid, y_valid)

import pandas as pd
import seaborn as sn
def plot_confusion_matrix(cm, model_name):
  df_cm = pd.DataFrame(
      cm,
      index=[artists_top_name.iloc[i].replace('_', ' ') for i in range(artists_top_name.shape[0] - 1)],
      columns=[artists_top_name.iloc[i].replace('_', ' ') for i in range(artists_top_name.shape[0] - 1)]
  )
  plt.figure(figsize=(15, 10))
  sns_plot = sn.heatmap(df_cm, annot=False)
  fig = sns_plot.get_figure()
  fig.savefig(f'{model_name}_confusionMatrix.png')


plot_confusion_matrix(cm, 'ResNet50')

vgg16_base_model = VGG16(weights='imagenet', include_top=False, input_shape=train_input_shape)
for layer in vgg16_base_model.layers:
    layer.trainable = True
# Add layers at the end
X_vgg = vgg16_base_model.output
X_vgg = Flatten()(X_vgg)

X_vgg = Dense(256, kernel_initializer='he_uniform')(X_vgg)
# X = Dropout(0.5)(X)
X_vgg = BatchNormalization()(X_vgg)
X_vgg = Activation('relu')(X_vgg)

X_resnet = Dense(128, kernel_initializer='he_uniform')(X_resnet)
# X = Dropout(0.5)(X)
X_vgg = BatchNormalization()(X_vgg)
X_vgg = Activation('relu')(X_vgg)

vgg_output = Dense(n_classes, activation='softmax')(X_vgg)

vgg_model = Model(inputs=vgg16_base_model.input, outputs=vgg_output)

optimizer = Adam(lr=0.0001)
vgg_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

n_epoch = 10
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           mode='auto', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='auto')

vgg_history1 = vgg_model.fit_generator(generator=train_generator,
                                       steps_per_epoch=STEP_SIZE_TRAIN,
                                       validation_data=valid_generator,
                                       validation_steps=STEP_SIZE_VALID,
                                       epochs=n_epoch,
                                       shuffle=True,
                                       verbose=1,
                                       callbacks=[reduce_lr],
                                       class_weight=class_weights
                                       )

# Freeze core ResNet layers and train again
for layer in vgg_model.layers:
    layer.trainable = False

for layer in vgg_model.layers[16:]:
    layer.trainable = True

optimizer = Adam(lr=0.0001)

vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 10
vgg_history2 = vgg_model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr, early_stop],
                              class_weight=class_weights
                             )

# Merge history1 and history2
vgg_history = {}
vgg_history['loss'] = vgg_history1.history['loss'] + vgg_history2.history['loss']
vgg_history['acc'] = vgg_history1.history['accuracy'] + vgg_history2.history['accuracy']
vgg_history['val_loss'] = vgg_history1.history['val_loss'] + vgg_history2.history['val_loss']
vgg_history['val_acc'] = vgg_history1.history['val_accuracy'] + vgg_history2.history['val_accuracy']
vgg_history['lr'] = vgg_history1.history['lr'] + vgg_history2.history['lr']

plt.style.use('seaborn')
plot_training(vgg_history, 'VGG16')

# Prediction accuracy on train data
vgg_score = vgg_model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on Training Data: ", vgg_score[1])

# Prediction accuracy on CV data
vgg_score = vgg_model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV Data: ", vgg_score[1])

p_valid, y_valid = get_predictions_and_labels(vgg_model, valid_generator, STEP_SIZE_VALID)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(p_valid, y_valid)
plot_confusion_matrix(cm, 'VGG16')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

custom_model = Sequential()
custom_model.add(Conv2D(32, (3, 3), input_shape=train_input_shape))
custom_model.add(Activation('relu'))
custom_model.add(MaxPooling2D(pool_size=(2, 2)))

custom_model.add(Conv2D(32, (3, 3)))
custom_model.add(Activation('relu'))
custom_model.add(MaxPooling2D(pool_size=(2, 2)))

custom_model.add(Conv2D(64, (3, 3)))
custom_model.add(Activation('relu'))
custom_model.add(MaxPooling2D(pool_size=(2, 2)))

custom_model.add(Flatten())
custom_model.add(Dense(64))
custom_model.add(Activation('relu'))
custom_model.add(Dropout(0.5))
custom_model.add(Dense(n_classes))
custom_model.add(Activation('softmax'))

custom_model.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

n_epoch = 10
custom_history = custom_model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                            validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                                            epochs=n_epoch,
                                            shuffle=True,
                                            verbose=1,
                                            callbacks=[reduce_lr, early_stop],
                                            class_weight=class_weights
                                            )

custom_history_dict = {}
custom_history_dict['loss'] = custom_history.history['loss']
custom_history_dict['acc'] = custom_history.history['accuracy']
custom_history_dict['val_loss'] = custom_history.history['val_loss']
custom_history_dict['val_acc'] = custom_history.history['val_accuracy']
custom_history_dict['lr'] = custom_history.history['lr']

plt.style.use('seaborn')
plot_training(custom_history_dict, 'Custom Deep CNN')

# Prediction accuracy on train data
custom_score = custom_model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on Training Data: ", custom_score[1])

# Prediction accuracy on CV data
vgg_score = custom_model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV Data: ", custom_score[1])

p_valid, y_valid = get_predictions_and_labels(custom_model, valid_generator, STEP_SIZE_VALID)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(p_valid, y_valid)
plot_confusion_matrix(cm, 'Custom CNN')

# put any url of the artwork to know the artist!
import imageio
import cv2

def predict_artist_from_web(model, url, model_name):
  web_image = imageio.imread(url)
  web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
  web_image = img_to_array(web_image)
  web_image /= 255.
  web_image = np.expand_dims(web_image, axis=0)


  prediction = model.predict(web_image)
  prediction_probability = np.amax(prediction)
  prediction_idx = np.argmax(prediction)
  print(f"Model: {model_name}")
  print(f"Predicted artist: {artists_top_name.iloc[prediction_idx].replace('_', ' ')}")
  print(f"Prediction score: {prediction_probability * 100}%", )

  plt.imshow(imageio.imread(url))
  plt.axis('off')
  plt.show()


  url = 'https://static01.nyt.com/images/2018/03/02/arts/design/02picasso-print/01picasso1-superJumbo.jpg'
  predict_artist_from_web(vgg_model, url, 'VGG16')

  url = 'https://static01.nyt.com/images/2018/03/02/arts/design/02picasso-print/01picasso1-superJumbo.jpg'
  predict_artist_from_web(resnet_model, url, 'ResNet50')

  url = 'https://static01.nyt.com/images/2018/03/02/arts/design/02picasso-print/01picasso1-superJumbo.jpg'
  predict_artist_from_web(custom_model, url, 'Custom CNN')

