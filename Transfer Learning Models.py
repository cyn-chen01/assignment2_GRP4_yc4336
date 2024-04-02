#Transfer Learning Models
#Densenet
pretrained_densenet = tf.keras.applications.DenseNet201(input_shape=(192, 192, 3), weights='imagenet', include_top=False)

for layer in pretrained_densenet.layers:
  layer.trainable = False

x1 = pretrained_densenet.output
x1 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x1)
x1 = tf.keras.layers.Flatten(name="flatten_head")(x1)
x1 = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x1)
x1 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x1)
model_out = tf.keras.layers.Dense(3, activation='softmax', name="predictions_head")(x1)

model_densenet = Model(inputs=pretrained_densenet.input, outputs=model_out)
model_densenet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
    )

history_densenet=model_densenet.fit(X_train, y_train, epochs = 15, validation_split= 0.2,callbacks=[lr_callback])

model_densenet.save("model_densenet.h5", save_format="h5")

# Reload model and data
model_densenet = tf.keras.models.load_model('model_densenet.h5',compile=False)
model_densenet.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy'])
model_densenet.summary()

#Efnet
!pip install efficientnet
import efficientnet.tfkeras as efn

# https://github.com/keras-team/keras/issues/9064

pretrained_efnet = efn.EfficientNetB7(input_shape=(192, 192, 3), weights='noisy-student', include_top=False)

for layer in pretrained_efnet.layers:
  layer.trainable = False

x2 = pretrained_efnet.output
x2 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x2)
x2 = tf.keras.layers.Flatten(name="flatten_head")(x2)
x2 = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x2)
x2 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x2)
model_out = tf.keras.layers.Dense(3, activation='softmax', name="predictions_head")(x2)

model_efnet = Model(inputs=pretrained_efnet.input, outputs=model_out)

model_efnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
    )
history_efnet=model_efnet.fit(X_train, y_train, epochs = 15, validation_split=0.2,callbacks=[lr_callback])

model_efnet.save("model_efnet.h5", save_format="h5")

# Reload model and data
model_efnet = tf.keras.models.load_model("model_efnet.h5",compile=False)
model_efnet.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy'])
model_efnet.summary()

#VGG
pretrained_vgg = tf.keras.applications.VGG16(input_shape=(192, 192, 3), weights='imagenet', include_top=False)

for layer in pretrained_vgg.layers:
  layer.trainable = False

x3 = pretrained_vgg.output
x3 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x3)
x3 = tf.keras.layers.Flatten(name="flatten_head")(x3)
x3 = tf.keras.layers.Dense(128, activation="relu", name="dense_head")(x3)
x3 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x3)
x3 = tf.keras.layers.Dense(64, activation="relu", name="dense_head_2")(x3)
x3 = tf.keras.layers.Dropout(0.5, name="dropout_head_2")(x3)
model_out = tf.keras.layers.Dense(3, activation='softmax', name="predictions_head")(x3)

model_vgg = Model(inputs=pretrained_vgg.input, outputs=model_out)

model_vgg.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
    )

history_vgg=model_vgg.fit(X_train, y_train, epochs = 15, validation_split=0.2,callbacks=[lr_callback])

model_vgg.save("model_vgg.h5", save_format="h5")

# Reload model and data
model_vgg = tf.keras.models.load_model('model_vgg.h5',compile=False)
model_vgg.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy'])
model_vgg.summary()