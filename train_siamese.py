import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from keras.callbacks import EarlyStopping
from classification_models_3D.tfkeras import Classifiers

BATCH_SIZE = 8
EPOCHS = 150

def decode_example(record_bytes) -> dict:
    example = tf.io.parse_example(
        record_bytes,     
        features = {
          'image_raw': tf.io.FixedLenFeature([], dtype=tf.string)
          }
    )
    return example

def parse_1_example(example):
    X = tf.io.parse_tensor(example['image_raw'], out_type=tf.float32)
    return tf.expand_dims(X, 3) 


def get_batched_dataset(files, batch_size = BATCH_SIZE):
    dataset = (
        tf.data.Dataset.list_files(files)
        .flat_map(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"))
        .map(decode_example)
        .map(parse_1_example)
        .batch(batch_size, drop_remainder=True)
    )
    return dataset

def get_dataset():
    file = open('./negative_images_w', 'rb')
    negative_images_w = pickle.load(file)
    
    # record names prefixes
    prefix = ["anchor", "positive", "negative"]
    records=[prefix[0] + '_000-of-000.tfrecords', 
             prefix[1] + '_000-of-000.tfrecords', 
             prefix[2] + '_000-of-000.tfrecords']
    weight_dataset = tf.data.Dataset.from_tensor_slices(negative_images_w)
    weight_dataset = weight_dataset.batch(BATCH_SIZE)

    anchor_dataset = get_batched_dataset(records[0])

    positive_dataset = get_batched_dataset(records[1])
    negative_dataset = get_batched_dataset(records[2])
    ds = tf.data.Dataset.zip((anchor_dataset,
            positive_dataset,
            negative_dataset, 
            weight_dataset))

    image_count = min(len(list(anchor_dataset)), 
                      len(list(positive_dataset)), 
                      len(list(negative_dataset)), 
                      len(weight_dataset))

    # split dataset to train and validation
    train_dataset = ds.take(round(image_count * 0.8))
    train_dataset = train_dataset.shuffle(buffer_size = 120)
    val_dataset = ds.skip(round(image_count * 0.8))
    return train_dataset, val_dataset

def get_model():
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

    inputs = keras.Input((160, 192, 192, 1))
    x = layers.Conv3D(filters=3, kernel_size=3, activation="relu", padding='same',
                     kernel_initializer=initializer)(inputs)
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_cnn = ResNet18(input_shape=(160, 192, 192, 3), weights='imagenet', include_top=False) 

    for layer in base_cnn.layers:
        layer.trainable = True
    base_cnn = base_cnn(x)

    dense = layers.Flatten()(base_cnn)
    dense = layers.Dense(64, activation="relu", kernel_initializer=initializer)(dense)
    dense = layers.BatchNormalization()(dense)

    dense = layers.Dense(32, activation="relu", kernel_initializer=initializer)(dense)
    dense = layers.BatchNormalization()(dense) 
    output = layers.Dense(8)(dense)
    embedding = Model(inputs, output, name="embedding")
    return embedding

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative, weight):
        dot_ap = tf.matmul(anchor, tf.transpose(positive))
        square_norm  = tf.linalg.diag_part(dot_ap)
        ap_distance = tf.expand_dims(square_norm, 0) - (2*dot_ap) + tf.expand_dims(square_norm, 1)
        
        dot_an = tf.matmul(anchor, tf.transpose(negative))
        square_norm  = tf.linalg.diag_part(dot_an)
        an_distance = tf.expand_dims(square_norm, 0) - (2*dot_an) + tf.expand_dims(square_norm, 1)
        
        # weighing start
        rate = 10 - weight
        rate = 1 + (rate/10.0)
        # weighing end
        
        an_distance = an_distance         
        return (ap_distance, an_distance)

class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        triplet_loss = ap_distance - an_distance  + self.margin
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        valid_triplets = tf.math.greater(triplet_loss, 0)
        loss = tf.reduce_sum(triplet_loss)/(tf.reduce_sum(tf.cast(valid_triplets, tf.float32))+1e-16)
        loss = tf.maximum(loss, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

def train_model(siamese_network):
    early_stopping_monitor = EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=200,
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=100,
        decay_rate=0.9)
    train_dataset, val_dataset = get_dataset()
    siamese_model = SiameseModel(siamese_network, margin = 1.)
    siamese_model.compile(optimizer=optimizers.Adam(learning_rate = lr_schedule))
    siamese_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, 
                                callbacks=[early_stopping_monitor])
    return siamese_model

def main():
    shape = (160, 192, 192, 1) 
    anchor_input = layers.Input(name="anchor", shape=shape) 
    positive_input = layers.Input(name="positive", shape=shape)
    negative_input = layers.Input(name="negative", shape=shape)
    weight = layers.Input(name="weight", shape=(()))
    
    embedding = get_model()
    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
        weight,
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input, weight], outputs=distances
    )
    trained_model = train_model(siamese_network)

if __name__ == "__main__":
    main()