import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn')

df = pd.read_csv("data/artists.csv")

df.drop(['bio', 'wikipedia', 'id'], axis=1, inplace=True)

df_year = pd.DataFrame(df.years.str.split().tolist(), columns = ['birth','-','death'])

df_year.drop('-', axis=1, inplace=True)

df["birth"] = df_year['birth']
df["death"] = df_year['death']
df.drop(["years"], axis=1, inplace=True)

df["birth"] = df["birth"].apply(lambda x: int(x))
df["death"] = df["death"].apply(lambda x: int(x))

df["age"] = df.death - df.birth

df['age'] = df['age']
bins=[30, 55, 65, 77, 98]
labels=["young adult", "early adult", "adult", "senior"]
df['age_group'] = pd.cut(df['age'], bins,labels=labels)

plt.figure(figsize=(18,5))

sns.barplot(x=df['nationality'].value_counts().index,y=df['nationality'].value_counts().values, label='Nationality')
plt.title('Nationality')
plt.xticks(rotation=75)
plt.ylabel('Rates')
plt.legend(loc=0)
plt.savefig('nationality.png')
plt.show()

plt.figure(figsize=(18,5))
sns.barplot(x=df['genre'].value_counts().index,
            y=df['genre'].value_counts().values)
plt.xlabel('genre')
plt.xticks(rotation=75)
plt.ylabel('Frequency')
plt.title('Show of genre Bar Plot')
plt.savefig('genre.png')
plt.show()

plt.figure(figsize=(22, 5))
sns.barplot(x="genre", y="paintings", hue="age_group", data=df)
plt.xticks(rotation=75)
plt.title('Genre')
plt.savefig('genre_with_paintings.png')
plt.show()

plt.figure(figsize=(17,8))
sns.barplot(x = "age_group", y = "paintings", hue = "genre", data = df)
plt.xticks(rotation=75)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('genre_with_age.png')
plt.show()

f,ax=plt.subplots(figsize=(9,10))
sns.barplot(x=df['age_group'].value_counts().values,y=df['age_group'].value_counts().index,alpha=0.5,color='red',label='age_group')
sns.barplot(x=df['genre'].value_counts().values,y=df['genre'].value_counts().index,color='blue',alpha=0.7,label='genre')
ax.legend(loc='upper right',frameon=True)
ax.set(xlabel='age_group , genre',ylabel='Groups',title="age_group vs genre ")
plt.show()

df['age'].unique()
len(df[(df['age'] > 50)].paintings)
f,ax1=plt.subplots(figsize=(25,10))
sns.pointplot(x=np.arange(1,41),y=df[(df['age']>50)].paintings,color='lime',alpha=0.8)

plt.xlabel('age>50 paintings')
plt.ylabel('Frequency')
plt.title('age>50 & paintings')
plt.xticks(rotation=90)
plt.grid()
plt.show()

sns.kdeplot(df['paintings'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('paintings Kde Plot System Analysis')
plt.savefig('save_fig_kde.png')
plt.show()
# plt.savefig('save_fig_kde.png')

ax = sns.distplot(df['age'])
plt.savefig('age_kde.png')
plt.show()


ax = sns.distplot(df['birth'])
plt.show()
#plt.savefig('birth_kde.png')

from keras.utils import np_utils
from keras_preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np


def convert_filenames_to_image_arrays(files):
  w, h = 256, 256
  images_as_array=[]
  for file in files:
    # print(file)
    # print(file)
    img = load_img(file)
    img = img.resize((w, h))
    img = img_to_array(img)
    img = img / 255.
    images_as_array.append(img)
  return np.array(images_as_array)

from sklearn.datasets import load_files

def load_dataset(path):
  data = load_files(path)
  files = np.array(data['filenames'])
  targets = np.array(data['target'])
  target_labels = np.array(data['target_names'])
  return files, targets, target_labels

x_train, Y_train, target_labels = load_dataset('data/images/images')
print('Loading complete.')


import matplotlib.pyplot as plt

indices = [np.random.randint(0, x_train.shape[0]) for i in range(25)]
random_files = [x_train[val] for val in indices]
labels = [target_labels[Y_train[val]] for val in indices]
images = convert_filenames_to_image_arrays(random_files)
plt.figure(figsize=(15, 15))
images = convert_filenames_to_image_arrays(random_files)
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.title(f'Artist: {labels[i]}')
  plt.imshow(images[i]);
plt.savefig('Artworks.png')