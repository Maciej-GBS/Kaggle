import tensorflow as tf
import pandas as pd
import numpy as np

def load_meta():
    return dict(pd.read_csv('meta_mean.csv').loc[0, :])

def load(path, meta, cols=['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket']):
    features = pd.read_csv(path, usecols=cols)
    def normalize_feature(nm):
        nparr = np.array(features[nm]) - meta[nm]
        return nparr / max(np.abs(nparr))
    features.loc[:, 'Age'] = normalize_feature('Age')
    features.loc[:, 'Name'] = normalize_feature('Name')
    features.loc[:, 'Ticket'] = normalize_feature('Ticket')
    return features

def load_dataset(meta):
    data = load('dataset/train.csv', meta)
    features = data.drop('Survived', axis=1)
    y_dash = data['Survived'].tolist()
    y_dash = pd.DataFrame({'Survived':y_dash, 'Missing':(np.array(y_dash) + 1) % 2})
    return features, y_dash

def load_test(meta):
    return load('dataset/test.csv', meta, ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket'])

def main():
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(7,))
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='titanic_model')
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse', metrics=['accuracy'])
    metadata = load_meta()
    x_train, y_train = load_dataset(metadata)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=1, validation_split=0.1)
    model.save('models/pre.h5')
    print("="*50)
    print(history.history)
    o = load_test(metadata)
    labels = model.predict(x=o)
    print(labels)

if __name__=='__main__':
    main()
