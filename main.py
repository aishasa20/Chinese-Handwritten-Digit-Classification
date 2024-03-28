
import random
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# load the dataset and create train and test data sets
dataset = pd.read_csv("/Users/aisha/Downloads/chineseMNIST (1).csv")

# Plot the count of each Chinese number
plt.hist(dataset['character'].values, bins=30)
plt.title('Count of Chinese Numbers')
plt.xlabel('Number')
plt.ylabel('Count')
plt.xticks([])
plt.yticks([])
plt.show()

# create a FontProperties object with the name of the Chinese font installed
chinese_font = FontProperties(fname="/Users/aisha/Downloads/SimHei.ttf")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
sb.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Arial']})


chinese_numbers_dict = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                        "百": 11, "千": 12, "万": 13, "亿": 14}

dataset.iloc[:, -2] = dataset.iloc[:, -2].replace(chinese_numbers_dict)

# Extract the feature data and labels from the DataFrame
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2]

# display 25 random images from training dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    random_indice = random.randint(0, 15000)
    pixel = dataset.iloc[random_indice, :-2].values
    pixel = pixel.reshape(64, 64)
    label = dataset.iloc[random_indice, -1]
    eng_label = dataset.iloc[random_indice, -2]
    pixel = pixel.astype(float) / 255.0
    plt.imshow(pixel, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(label, fontproperties=chinese_font)
    plt.xlabel(eng_label)
plt.tight_layout()
plt.show()

# scale the data set
X = X / 255

# Partition the dataset into train and test sets. Print the shapes of the train and test data sets.
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)

# Reshaping X_train and X_test
X_train = np.reshape(X_train, (-1, 64, 64))
X_test = np.reshape(X_test, (-1, 64, 64))

# create keras model of sequence of layers
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(64, 64)))
model.add(keras.layers.Dense(256, activation='relu', ))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="softmax"))

# display the model's summary
model.summary()

# set the loss and compile the function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
              )

training_loss = model.fit(X_train, y_train, epochs=25, verbose=1)
testing_loss = model.fit(X_test, y_test, epochs=25, verbose=1)

# plot the loss curve
training_df = pd.DataFrame(training_loss.history)
testing_df = pd.DataFrame(testing_loss.history)
training_df.plot()
testing_df.plot()
plt.show()

# predict labels
y_pred = model.predict(X_test)

# Convert the predicted labels to their corresponding Chinese characters
y_pred_chars = []
for i in range(len(y_pred)):
    y_pred_chars.append(list(chinese_numbers_dict.keys())[list(chinese_numbers_dict.values()).index(int(y_pred[i]))])

# Convert the actual labels to their corresponding Chinese characters
y_test_chars = []
for i in range(len(y_test[:30])):
    y_test_chars.append(list(chinese_numbers_dict.keys())[list(chinese_numbers_dict.values()).index(int(y_test.iloc[i]))])

# Visualize and predict actual image labels for first 30 images
plt.figure(figsize=[10, 10])
for i in range(30):
    plt.subplot(6, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Actual: {}\nPredicted: {}".format(y_test_chars[i], y_pred_chars[i]), fontproperties=chinese_font)
plt.tight_layout
plt.show()

# compare predicted labels with actual labels
failed_indices = np.where(y_pred != y_test)[0]
failed_indices = np.random.choice(failed_indices, size=30, replace=False)

# display the misclassified images
plt.figure(figsize=(10, 10))
for i, index in enumerate(failed_indices):
    plt.subplot(6, 5, i + 1)
    plt.imshow(X_test[index], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Actual: {y_test.iloc[index]} \nPredicted: {y_pred[index]}", fontproperties=chinese_font)
plt.tight_layout()
plt.show()
