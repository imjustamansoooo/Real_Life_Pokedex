import pandas as pd
import pygame
import tensorflow as tf
from pathlib import Path
import json
import numpy as np
from PIL import Image


# =====================
# CONFIG
# =====================
MODEL_DIR = Path("pokemon-dataset-1000")
MODEL_PATH = MODEL_DIR / "pokemon_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

#IMAGE_PATH = Path("pokemon-dataset-1000/test/snorlax_12.png")
IMAGE_PATH = Path("test-images/967985ca22b2b00b485eca2bd47295d2.png")   # <-- change if needed
IMG_SIZE = (224, 224)

# =====================
# LOAD MODEL
# =====================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# =====================
# LOAD CLASS NAMES
# =====================
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} Pokémon classes.")

# =====================
# LOAD & PREPROCESS IMAGE
# IMPORTANT: NO MANUAL NORMALIZATION
# =====================
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize(IMG_SIZE)

img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

# =====================
# PREDICT
# =====================
predictions = model.predict(img_array)[0]

# Top prediction
class_id = int(np.argmax(predictions))
confidence = float(predictions[class_id])

# =====================
# OUTPUT
# =====================
print("\n===== POKEDEX RESULT =====")
print(f"Pokémon:   {class_names[class_id]}")
print(f"Confidence: {confidence:.2%}")

# Optional: Top 5 predictions (HIGHLY useful)
print("\nTop 5 predictions:")
top5 = predictions.argsort()[-5:][::-1]
for i in top5:
    print(f"{class_names[i]:15s}  {predictions[i]:.2%}")

print("==========================\n")

print("==========================")

file = "Pokédex_Info.xlsx"
pokedex = pd.read_excel(file)

found_pokemon = pokedex[pokedex["Pokemon Name"] == str(class_names[class_id]).capitalize()]
pokemon = found_pokemon.to_dict()

for i in pokemon:
    print(f"{i}:")
    value = next(iter(pokemon[i].values()))
    print(f"{value}\n")

print("==========================")
#
# # =====================
# # CONFIG
# # =====================
# DATASET_DIR = Path(__file__).parent / "pokemon-dataset-1000/dataset"
# MODEL_DIR = Path("pokemon-dataset-1000")
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 10
# SEED = 123
#
# MODEL_DIR.mkdir(exist_ok=True)
#
# # =====================
# # LOAD DATASET
# # =====================
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )
#
# class_names = train_ds.class_names
# num_classes = len(class_names)
#
# print(f"Loaded {num_classes} Pokémon classes")
#
# # =====================
# # SAVE CLASS NAMES (IMPORTANT)
# # =====================
# with open(MODEL_DIR / "class_names.json", "w") as f:
#     json.dump(class_names, f)
#
# print("Class names saved")
#
# # =====================
# # PERFORMANCE OPTIMIZATION
# # =====================
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# # =====================
# # DATA AUGMENTATION
# # =====================
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.1),
#     tf.keras.layers.RandomZoom(0.1),
#     tf.keras.layers.RandomContrast(0.1),
# ])
#
# # =====================
# # BASE MODEL (TRANSFER LEARNING)
# # =====================
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights="imagenet"
# )
#
# base_model.trainable = False
#
# # =====================
# # BUILD MODEL
# # =====================
# model = tf.keras.Sequential([
#     data_augmentation,
#     tf.keras.layers.Rescaling(1./255),
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(num_classes, activation="softmax")
# ])
#
# # =====================
# # COMPILE
# # =====================
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# # =====================
# # TRAIN
# # =====================
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS
# )
#
# # =====================
# # SAVE MODEL (IMPORTANT)
# # =====================
# model.save(MODEL_DIR / "pokemon_model.keras")
#
# print("Model saved successfully")


# print(pokedex["National\nDex"][user_input])
# print(pokedex["Pokemon\nName"][user_input])
# print(pokedex["Type"][user_input])
# print(pokedex["HP"][user_input])
# print(pokedex["Attack"][user_input])
# print(pokedex["Defense"][user_input])
# print(pokedex["Special\nAttack"][user_input])
# print(pokedex["Special\nDefense"][user_input])
# print(pokedex["Speed"][user_input])
# print(pokedex["Total"][user_input])
# print(pokedex["Type I"][user_input])
# print(pokedex["Type II"][user_input])
# print(pokedex["Ability I"][user_input])
# print(pokedex["Ability II"][user_input])
# print(pokedex["Hidden Ability"][user_input])
# print(pokedex["Height (m)"][user_input])
# print(pokedex["Weight (lbs)"][user_input])
# print(pokedex["Weight (kg)"][user_input])
# print(pokedex["HP EVs"][user_input])
# print(pokedex["Attcak EVs"][user_input])
# print(pokedex["Defense EVs"][user_input])
# print(pokedex["Special Attack EVs"][user_input])
# print(pokedex["Special Defense EVs"][user_input])
# print(pokedex["Speed EVs"][user_input])
# print(pokedex["Experience\nValue"][user_input])
# print(pokedex["Catch Rate"][user_input])
# print(pokedex["Experience\nGrowth"][user_input])
# print(pokedex["Base Friendship"][user_input])
# print(pokedex["pokedex Color"][user_input])
# print(pokedex["Gender Ratio"][user_input])
# print(pokedex["Egg Cycles"][user_input])
# print(pokedex["Egg Group I"][user_input])
# print(pokedex["Egg Group II"][user_input])
# print(pokedex["Evolution\nMethod"][user_input])
