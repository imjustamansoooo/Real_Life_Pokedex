import pandas as pd
import pygame
import tensorflow as tf
from pathlib import Path
import json
import numpy as np
from PIL import Image
import time
import random
import os

ConsoleLoop = True
top_confidence = float(0.90)

# =====================
# CONFIG
# =====================
MODEL_DIR = Path("pokemon-dataset-1000")
MODEL_PATH = MODEL_DIR / "pokemon_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

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

file = "Pokédex_Info.xlsx"
pokedex = pd.read_excel(file)
print("Loaded Excel Pokédex Data")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def capture():
    Top5 = {}
    photoId = random.randint(1, 100000)
    os.system(f"fswebcam -r 1280x720 --jpeg 85 -D 1 captures/{photoId}.jpg")
    IMAGE_PATH = Path(f"captures/{photoId}.jpg")  # <-- change if needed
    IMG_SIZE = (224, 224)
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
    confidence_final = float(f"{confidence:.2}")

    # =====================
    # OUTPUT
    # =====================
    if top_confidence > confidence_final:
        count2_5 = 0
        top5 = predictions.argsort()[-5:][::-1]
        for i in top5:
            Top5[str(f"{class_names[i]:15s}".strip())] = f"{predictions[i]:.2}%"

        print("Which Pokémon is it?")
        for o in Top5:
            count2_5 = count2_5 + 1
            print(f"[{count2_5}]: {o}")

        pokemonName = str(input("_: ").lower().strip())

    else:
        print("\n===== POKEDEX RESULT =====")
        print(f"Pokémon:   {class_names[class_id]}")
        print(f"Confidence: {confidence_final}%")
        pokemonName = str(class_names[class_id])

    print("==========================\n")

    print("==========================")

    try:
        os.mkdir(f"captures/pokemonCaptures/{pokemonName}")
        print(f"Directory for {pokemonName} created\n")
    except FileExistsError:
        print(f"Directory for {pokemonName} found\n")
    except Exception as e:
        print(f"An error occurred: {e}\n")

    os.rename(f"captures/{photoId}.jpg", f"captures/pokemonCaptures/{pokemonName}/[{pokemonName}]_({photoId}).jpg")

    found_pokemon = pokedex[pokedex["Pokemon Name"] == pokemonName.capitalize()]
    pokemon = found_pokemon.to_dict()

    for i in pokemon:
        print(f"{i}:")
        value = next(iter(pokemon[i].values()))
        print(f"{value}\n")

    print("==========================")


def search():
    poke_search = input("(Search)_: ").capitalize().strip()
    
    print("==========================")
    try:
        if poke_search.isdigit():
            found_pokemon = pokedex[pokedex["National Dex"] == float(poke_search)]
        else:
            found_pokemon = pokedex[pokedex["Pokemon Name"] == poke_search]
        pokemon = found_pokemon.to_dict()

        for i in pokemon:
            print(f"{i}:")
            value = next(iter(pokemon[i].values()))
            print(f"{value}\n")

    except:
        print("Pokêmon Not Found!")

    print("==========================")


def central_control(cmdA):
    for y in commands:
        if y == cmdA:
            return commands[y]()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


commands = {"search": search, "capture": capture}

print("WELCOME TO YOUR POKEDEX\n")

while ConsoleLoop:
    console = str(input("_: ").lower().strip())

    if console == "quit":
        quit()
    else:
        central_control(console)



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
