import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
import numpy as np

# Load files
data = np.load('datasets/data0.npz')
states = data["states"]
trumpInfo = data["trumpInfo"]
rewards = data["rewards"]

print(states.shape)
print(trumpInfo.shape)
print(rewards.shape)


def model_card():
    model = models.Sequential()
    model.add(layers.Conv2D(filters = 32, kernel_size = (2,2) ,activation='relu', input_shape= (8,4,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))

    # model = Input((8,4,1))
    # model = layers.Conv2D(filters = 32, kernel_size = (2,2) ,activation='relu')(model)
    # model = layers.MaxPooling2D((2, 2))(model)
    # model = layers.Conv2D(64, (1, 1), activation='relu')(model)
    # model = layers.Flatten()(model)
    # model = layers.Dense(64, activation='relu')(model)
    # model = layers.Dense(10, activation='relu')(model)

    return model

def model_trump():
    model = models.Sequential()
    model.add(layers.Dense(input_shape = (5,), activation='relu', units=1))
    model.add(layers.Dense(10, activation='relu'))
    # model = Input((5,))
    # model = layers.Dense(10, activation='relu')(model)
    return model

def get_model():
    model1 = model_card()
    # model2 = model_card()
    # model3 = model_card()
    model4 = model_trump()
    # print(model3.layers[-1])
    # print(model3.summary())
    # merged = layers.Concatenate()([model1.output, model2.output, model3.output, model4.output])
    merged = layers.Concatenate()([model1.output, model4.output]) # use only two model
    output = layers.Dense(10, activation='relu')(merged)
    output = layers.Dense(1, activation='linear')(output)

    # final_model = Model(inputs = [model1.input, model2.input, model3.input, model4.input], outputs = [output])
    final_model = Model(inputs = [model1.input, model4.input], outputs = [output])

    return final_model



def train_play_model():
    final_model = get_model()
    final_model.summary()

    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.Huber(delta = 0.1),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # final_model = models.load_model('compressed_model_normalized.h5')

    final_model.fit(
        x=[states,trumpInfo],
        y=rewards,
        batch_size=1000,
        epochs=500,
        verbose=2,
        shuffle=True,
    )

    final_model.save('model/model0.h5')


if __name__ == "__main__":
    train_play_model()

