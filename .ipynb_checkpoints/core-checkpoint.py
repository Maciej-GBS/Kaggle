import tensorflow as tf
import pandas as pd
import numpy as np

def load_meta():
    return dict(pd.read_csv('meta_mean.csv').loc[0, :])

def load(path, meta, cols=['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket']):
    features = pd.read_csv(path, usecols=cols)
    for nom in ['Name','Age','SibSp','Parch','Ticket']:
        if nom in list(features.columns):
            nparr = np.array(features[nom]) - meta[nom]
            features.loc[:, nom] = nparr / max(np.abs(nparr))
    if 'Pclass' in list(features.columns):
        classe = pd.DataFrame({'1st':np.where(features['Pclass']==1,1,0), '2nd':np.where(features['Pclass']==2,1,0), '3rd':np.where(features['Pclass']==3,1,0)})
        features = features.drop('Pclass', axis=1)
        features = pd.concat([features, classe], axis=1)
    return features

def load_dataset(meta):
    data = load('dataset/train.csv', meta)
    data.to_csv('internal.csv', index=False)
    features = data.drop('Survived', axis=1)
    y_dash = data['Survived'].tolist()
    y_dash = pd.DataFrame({'Survived':y_dash, 'Missing':(np.array(y_dash) + 1) % 2})
    return features, y_dash

def load_test(meta):
    return load('dataset/test.csv', meta, ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket'])

def main():
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(9,))
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='titanic_model')
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    metadata = load_meta()
    x_train, y_train = load_dataset(metadata)
    print(x_train)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=100, validation_split=0.1)
    model.save('models/pre.h5')
    print("="*50)
    print(history.history['acc'])
    o = load_test(metadata)
    labels = model.predict(x=o)
    labels = pd.DataFrame(labels, columns=['S', 'M'])
    labels = pd.Series(np.where(labels['S'] > labels['M'], 1, 0))
    result = load('dataset/test_id.csv', metadata, ['PassengerId'])
    result = pd.concat([result, labels], axis=1)
    result.rename(columns={0:'Survived'}, inplace=True)
    print(result)
    result.to_csv('output.csv', index=False)

if __name__=='__main__':
    main()
