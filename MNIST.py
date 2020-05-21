from keras.datasets import mnist 
dataset=mnist.load_data('mnist.db')            #download mnist dataset from keras datasets
train,test=dataset                                                       #dividing dataset in train and test
X_train,y_train=train                                               #dividing train set in x and y ;x is image and y is label
X_test,y_test=test
X_train_1d=X_train.reshape(-1,28*28)           #we are  flatting the images
X_test_1d=X_test.reshape(-1,28*28)
X_train = X_train_1d.astype('float32')           #converting in float32 because most of the operations in neural network are continuous and not discrete
X_test = X_test_1d.astype('float32') 
from keras.utils.np_utils import to_categorical               #y is having 10 categories so we are converting in one hot vector
y_train_cat = to_categorical(y_train)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()                                                                                       # we are creating fully packed layer
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=256, input_dim=28*28, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
from keras.optimizers import Adam
model.compile(optimizer=Adam(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
 model.fit(X_train, y_train_cat, epochs=20)