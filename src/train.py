import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model

def get_model(df, categorical_cols):
    inputs = []
    outputs = []

    for c in categorical_cols:
        num_unique_vals = int(df[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_vals / 2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_vals + 1, embed_dim, name=c)(inp)
        # apply dropout
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        inputs.append(inp)
        outputs.append(out)

    x = layers.Concatenate()(outputs)
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    y = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=y)

    return model

if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    sample = pd.read_csv("../input/sample_submission.csv")

    test.loc[:, "target"] = -1
    data = pd.concat([train, test]).reset_index(drop=True)
    #print(data.shape)
    #print(train.shape)
    #print(test.shape)

    features = [
        f for f in train.columns if f not in ["id", "target"]
    ]
    #print(features)

    for feat in features:
        lbl = preprocessing.LabelEncoder()
        data.loc[:, feat] = lbl.fit_transform(data[feat].astype(str).fillna("-1").values)
    #print(data.head)

    train = data[data.target != -1].reset_index(drop=True)
    test = data[data.target == -1].reset_index(drop=True)

    #get_model(train, features).summary()
    model = get_model(train, features)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit([train.loc[:, f].values for f in features], train.target.values)

