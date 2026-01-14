# species_train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# CONFIG
TRAIN_DIR = "dataset/species/train"
VAL_DIR   = "dataset/species/val"
IMG_SIZE = (224,224)
BATCH = 32
EPOCHS = 1
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# DATAGEN
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   zoom_range=0.1)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

# MODEL
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(base.input, out)

# freeze base
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# TRAIN
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
# Save trained species model
model.save("species_model.keras")
print("✅ species_model.keras saved successfully")
