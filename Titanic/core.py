import tensorflow as tf
import pandas as pd
import numpy as np

META = pd.DataFrame()

def load(path, cols=['Survived','Name','Sex','Age','SibSp','Parch','Ticket','Fare']):
    global META
    features = pd.read_csv(path, usecols=cols)
    if META.empty:
        META = features.describe()
    for nom in ['Name','Age','SibSp','Parch','Ticket','Fare']:
        if nom in list(features.columns):
            nparr = np.array(features[nom]) - META[nom]['mean']
            features.loc[:, nom] = nparr / META[nom]['std']
    if 'Pclass' in list(features.columns):
        classe = pd.DataFrame({'1st':np.where(features['Pclass']==1,1,0), '2nd':np.where(features['Pclass']==2,1,0), '3rd':np.where(features['Pclass']==3,1,0)})
        features = features.drop('Pclass', axis=1)
        features = pd.concat([features, classe], axis=1)
    if 'Age' in list(features.columns):
        features = pd.concat([features, pd.DataFrame({'Age2': np.square(np.array(features['Age']))})], axis=1)
    return features

def load_dataset():
    data = load('dataset/train_g.csv')
    data.to_csv('internal.csv', index=False)
    features = data.drop('Survived', axis=1)
    y_dash = data['Survived'].tolist()
    y_dash = pd.DataFrame({'Survived':y_dash, 'Missing':(np.array(y_dash) + 1) % 2})
    return features, y_dash

def load_test():
    return load('dataset/test.csv', ['Name','Sex','Age','SibSp','Parch','Ticket','Fare'])

def main():
    global META
    # Setup model
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(8,))
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='titanic_model')
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001), loss='mse', metrics=['accuracy'])
    model.load_weights('models/pre.h5')
    META = pd.read_csv('models/pre.meta', index_col=0)
    # Train model
    x_train, y_train = load_dataset()
    print(x_train)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=500, validation_split=0.1)
    model.save('models/pre.h5')
    META.to_csv('models/pre.meta')
    print("$"*50)
    print('Accuracy list', history.history['acc'])
    print("$"*50)
    # Get output on test dataset
    o = load_test()
    labels = model.predict(x=o)
    labels = pd.DataFrame(labels, columns=['S', 'M'])
    labels = pd.Series(np.where(labels['S'] > labels['M'], 1, 0))
    result = load('dataset/test_id.csv', ['PassengerId'])
    result = pd.concat([result, labels], axis=1)
    result.rename(columns={0: 'Survived'}, inplace=True)
    print(result)
    result.to_csv('output.csv', index=False)
    print("$"*50)
    # Recall and Precision on train.csv
    x_test = load('dataset/train.csv')
    labels = model.predict(x=x_test.drop('Survived', axis=1))
    labels = pd.DataFrame(labels, columns=['S', 'M'])
    labels = pd.Series(np.where(labels['S'] > labels['M'], 1, 0))
    result = pd.concat([x_test, labels], axis=1)
    result.rename(columns={0: 'Predicted'}, inplace=True)
    print(result)
    correct_result = np.asarray(result['Survived']==result['Predicted'])
    true_positives = float(np.sum(result.loc[correct_result, 'Survived']))
    correct_result = np.sum(correct_result)
    accuracy = correct_result / len(result)
    precision = true_positives / np.sum(result['Predicted'])
    recall = true_positives / np.sum(result['Survived'])
    print("Accuracy:", accuracy*100, "% | Precision:", precision*100, "% | Recall:", recall*100, "%")

if __name__=='__main__':
    main()
