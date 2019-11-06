import tensorflow as tf
import pandas as pd
import numpy as np

def load_meta():
    return dict(pd.read_csv('meta_mean.csv').loc[0, :])

def load(path, meta, cols=['Survived','Name','Sex','Age','SibSp','Parch','Ticket','Fare']):
    features = pd.read_csv(path, usecols=cols)
    for nom in ['Name','Age','SibSp','Parch','Ticket','Fare']:
        if nom in list(features.columns):
            nparr = np.array(features[nom]) - meta[nom]
            features.loc[:, nom] = nparr / max(np.abs(nparr))
    if 'Pclass' in list(features.columns):
        classe = pd.DataFrame({'1st':np.where(features['Pclass']==1,1,0), '2nd':np.where(features['Pclass']==2,1,0), '3rd':np.where(features['Pclass']==3,1,0)})
        features = features.drop('Pclass', axis=1)
        features = pd.concat([features, classe], axis=1)
    if 'Age' in list(features.columns):
        features = pd.concat([features, pd.DataFrame({'Age2': np.square(np.array(features['Age']))})], axis=1)
    return features

def load_dataset(meta):
    data = load('dataset/train_g.csv', meta)
    data.to_csv('internal.csv', index=False)
    features = data.drop('Survived', axis=1)
    y_dash = data['Survived'].tolist()
    y_dash = pd.DataFrame({'Survived':y_dash, 'Missing':(np.array(y_dash) + 1) % 2})
    return features, y_dash

def load_test(meta):
    return load('dataset/test.csv', meta, ['Name','Sex','Age','SibSp','Parch','Ticket','Fare'])

def main():
    # Setup model
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(8,))
    x = tf.keras.layers.Dense(16)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='titanic_model')
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    metadata = load_meta()
    # Train model
    x_train, y_train = load_dataset(metadata)
    print(x_train)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=2, validation_split=0.1)
    model.save('models/pre.h5')
    print("$"*50)
    print('Accuracy list', history.history['acc'])
    print("$"*50)
    # Get output on test dataset
    o = load_test(metadata)
    labels = model.predict(x=o)
    labels = pd.DataFrame(labels, columns=['S', 'M'])
    labels = pd.Series(np.where(labels['S'] > labels['M'], 1, 0))
    result = load('dataset/test_id.csv', metadata, ['PassengerId'])
    result = pd.concat([result, labels], axis=1)
    result.rename(columns={0: 'Survived'}, inplace=True)
    print(result)
    result.to_csv('output.csv', index=False)
    print("$"*50)
    # Recall and Precision on train.csv
    x_test = load('dataset/train.csv', metadata)
    labels = model.predict(x=x_test.drop('Survived', axis=1))
    labels = pd.DataFrame(labels, columns=['S', 'M'])
    labels = pd.Series(np.where(labels['S'] > labels['M'], 1, 0))
    result = pd.concat([x_test, labels], axis=1)
    result.rename(columns={0: 'Predicted'}, inplace=True)
    print(result)
    correct_result = np.asarray(result['Survived']==result['Predicted'])
    true_positives = float(np.sum(result.loc[correct_result, 'Survived']))
    correct_result = np.sum(correct_result)
    positives = np.sum(result['Predicted'])
    precision = true_positives / correct_result
    recall = true_positives / positives
    print("Precision:", precision, "| Recall:", recall)

if __name__=='__main__':
    main()
