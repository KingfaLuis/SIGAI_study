

```
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10 #分类数
epochs = 12 # 训练轮数

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    #使用Theano的顺序
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    #使用tensorflow的顺序
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#将向量转化为二进制类矩阵
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#构建模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#全连接层
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#模型训练
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#模型评估，输出测试集的损失值和准确率
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Using TensorFlow backend.
    

    x_train shape: (60000, 28, 28, 1)
    60000 train samples
    10000 test samples
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/12
    60000/60000 [==============================] - 1648s 27ms/step - loss: 0.2753 - acc: 0.9157 - val_loss: 0.0612 - val_acc: 0.9799
    Epoch 2/12
    60000/60000 [==============================] - 1605s 27ms/step - loss: 0.0932 - acc: 0.9721 - val_loss: 0.0437 - val_acc: 0.9848
    Epoch 3/12
    60000/60000 [==============================] - 1656s 28ms/step - loss: 0.0718 - acc: 0.9791 - val_loss: 0.0441 - val_acc: 0.9861
    Epoch 4/12
    60000/60000 [==============================] - 1631s 27ms/step - loss: 0.0607 - acc: 0.9824 - val_loss: 0.0387 - val_acc: 0.9872
    Epoch 5/12
    60000/60000 [==============================] - 1650s 27ms/step - loss: 0.0542 - acc: 0.9839 - val_loss: 0.0351 - val_acc: 0.9881
    Epoch 6/12
    60000/60000 [==============================] - 1547s 26ms/step - loss: 0.0482 - acc: 0.9855 - val_loss: 0.0371 - val_acc: 0.9873
    Epoch 7/12
    60000/60000 [==============================] - 1620s 27ms/step - loss: 0.0449 - acc: 0.9871 - val_loss: 0.0360 - val_acc: 0.9885
    Epoch 8/12
    60000/60000 [==============================] - 1602s 27ms/step - loss: 0.0422 - acc: 0.9871 - val_loss: 0.0337 - val_acc: 0.9892
    Epoch 9/12
    60000/60000 [==============================] - 1599s 27ms/step - loss: 0.0391 - acc: 0.9879 - val_loss: 0.0379 - val_acc: 0.9892
    Epoch 10/12
    60000/60000 [==============================] - 1659s 28ms/step - loss: 0.0370 - acc: 0.9886 - val_loss: 0.0314 - val_acc: 0.9899
    Epoch 11/12
    60000/60000 [==============================] - 1649s 27ms/step - loss: 0.0362 - acc: 0.9887 - val_loss: 0.0339 - val_acc: 0.9899
    Epoch 12/12
    60000/60000 [==============================] - 1578s 26ms/step - loss: 0.0359 - acc: 0.9896 - val_loss: 0.0323 - val_acc: 0.9899
    Test loss: 0.03234027758775046
    Test accuracy: 0.9899
    


```
#保存模型
from keras.models import save_model,load_model

def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    new_model = load_model(fname)
    os.remove(fname)

    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test that new updates are the same with both models
    # 检测保存的模型和新定义的模型是否一致
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    new_model.train_on_batch(x, y)
    out = model.predict(x)
    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

```


```
json_string = model.to_json()
# json_string = model.to_yaml()
```


```
#加载模型
from keras.models import model_from_json

model1 = model_from_json(json_string)
```
