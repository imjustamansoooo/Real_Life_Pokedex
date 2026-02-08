import pandas as pd
import pygame
import tensorflow as tf
from pathlib import Path

# file = "Pokédex_Info.xlsx"
# pokedex = pd.read_excel(file)
#
# print("Welcome to the Pokédex!")
# user = float(input(":_ "))
#
# found_pokemon = pokedex[pokedex["National Dex"] == user]
# pokemon = found_pokemon.to_dict()
#
# for i in pokemon:
#     print(f"{i}:")
#     value = next(iter(pokemon[i].values()))
#     print(f"{value}\n")


Dataset_Dir = Path(__file__).parent / "pokemon-dataset-1000/dataset"

img_size = (224, 224)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    Dataset_Dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    Dataset_Dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)













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
