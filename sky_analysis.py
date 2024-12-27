from __future__ import annotations

import os
import logging
import math
import numpy as np
import keras

from datetime import datetime
from pprint import pprint
from paho.mqtt import client as mqtt_client
from PIL import Image, ImageOps  # Install pillow instead of PIL

# TODO: use this
#from pysolar.solar import *

from config import settings

VERSION = "0.1.0"
DEBUG = False

logger = logging.getLogger()

def connect_mqtt(host: string, port: number, client_id: string, username: string, password: string):
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
        else:
            logger.Error("Failed to connect, return code %d\n", rc)
            exit(1)

    # Set Connecting Client ID
    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)

    if username and password:
        client.username_pw_set(username, password)

    client.on_connect = on_connect

    logger.info(f"Connecting to MQTT broker at {host}:{port}")
    client.connect(host, port)

    return client

def subscribe(client: mqtt_client, topic: string, image_callback):
    client.subscribe(topic)
    client.on_message = image_callback


def setup_logging():
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filename=os.path.join(settings.datadir, 'ml_sky_analysis.log'), mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    logger.setLevel(logging.INFO)

    logger.info("ML Sky Analysis starting...")

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

def setup_datadir():
    if not os.path.exists(settings.datadir):
        os.makedirs(settings.datadir)

def load_model():
    # Load the labels
    image_classes = []
    
    logger.info("Loading model classes")
    with open(os.path.join(settings.datadir, settings.model_labels), "r") as f:
        image_classes = [l.rstrip() for l in f.readlines()]

    logger.info("Loading keras model")
    model = keras.models.load_model(os.path.join(settings.datadir, settings.model_data), compile=False)

    return model, image_classes

def analyze_image(model, images_classes, imageBytes):
    start_time = datetime.now()
    image_size = (256, 256)

    # determined by the first position in the shape tuple, in this case 1
    #data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)


    original_file = os.path.join(settings.datadir, "ccd1_20241223_214213.jpg")
    # original_file = os.path.join(settings.datadir, "original.jpg")
    cropped_file = os.path.join(settings.datadir, "cropped.jpg")

    # write the original file received from mqtt
    # with open(original_file, "wb") as f:
    #   f.write(imageBytes)

    # load the original file we just write. this could likely be done all in memory without
    # first saving the original file, but i kinda like having the original to see for testing
    # purposes and i doubt the performance impact could be that big from the extra write/read
    # operation. so leaving it this way
    original_image = Image.open(original_file)

    # get the orignial dimensions
    w, h = original_image.size
    cropped_image = original_image.crop((settings.crop_l, settings.crop_t, w-settings.crop_r, h-settings.crop_b))
    cropped_image.save(cropped_file)

    crop_w, crop_h = cropped_image.size

    crop_each_direction = int(math.sqrt(settings.crop_sections))

    tile_w = crop_w / crop_each_direction
    tile_h = crop_h / crop_each_direction

    total_tiles = settings.crop_sections - len(settings.discard_crop_sections)

    # an okta is defined as how many 8th's of the sky is covered by clouds, so since
    # we are dividing into multiple parts and classifying by how dense the clouds are
    # in each tile, we need to create a factor based on the total number of crop sections
    okta_factor = 8 / total_tiles

    tiles = []
    tile_results = []
    tile_index = 0
    total_oktas = 0
    total_weighted_oktas = 0
    # cloudy_count = 0

    # crop the tiles
    for y in range (0, crop_each_direction):
        for x in range(0, crop_each_direction):
            x_offset = x * tile_w
            y_offset = y * tile_h

            tiles.append(cropped_image.crop((x_offset, y_offset, x_offset+tile_w, y_offset+tile_h)))

    # loop through the tiles and perform inference on each
    for tile_index in range(0, settings.crop_sections):

        # skip discarded tiles
        if tile_index in settings.discard_crop_sections:
            continue

        tile = tiles[tile_index]
        tile_path = os.path.join(settings.datadir, f"tile_{tile_index}.jpg")
        tile.save(tile_path)

        ## keras v3
        image = tile.resize(image_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        top_score_index = np.argmax(prediction)
        image_class = images_classes[top_score_index]
        confidence_score = float(prediction[0][top_score_index])

        clear_night_confidence = float(prediction[0][0])
        light_clouds_night_confidence = float(prediction[0][1])
        medium_clouds_night_confidence = float(prediction[0][2])
        thick_clouds_night_confidence = float(prediction[0][3])

        total_weighted_oktas += light_clouds_night_confidence * 0.5 * okta_factor
        total_weighted_oktas += medium_clouds_night_confidence * 0.75 * okta_factor
        total_weighted_oktas += thick_clouds_night_confidence * 1 * okta_factor


        ## keras v2
        # tile_rgb = tile.convert("RGB")

        # # resizing the image to be at least 224x224 and then cropping from the center
        # tile_resized = ImageOps.fit(tile_rgb, image_size, Image.Resampling.LANCZOS)

        # # turn the image into a numpy array
        # image_array = np.asarray(tile_resized)

        # # Normalize the image
        # normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # # Load the image into the array
        # data[0] = normalized_image_array

        # # Predicts the model
        # prediction = model.predict(data, verbose=2)
        # index = np.argmax(prediction)
        # image_class = images_classes[index]
        # confidence_score = prediction[0][index]

        # Print prediction and confidence score
        if DEBUG:
            pprint(prediction, depth=4)
            print(f"Tile {tile_index:02} result: {image_class.ljust(6)} / {round(confidence_score * 100, 2)}% confidence")

        okta_score = 0

        if 'light_clouds' in image_class:
            okta_score = 0.5
        elif 'medium_clouds' in image_class:
            okta_score = 0.75
        elif 'thick_clouds' in image_class:
            okta_score = 1

        total_oktas += (okta_score * okta_factor)

        tile_results.append({
            "tile": tile_index,
            "class": image_class,
            "confidence": round(confidence_score, 4)
        })


    total_oktas = int(round(total_oktas, 0))
    total_weighted_oktas = int(round(total_weighted_oktas, 0))
    
    if total_oktas < 0:
        total_oktas = 0
    elif total_oktas > 8:
        total_oktas = 8

    cloudy_pct = total_oktas / 8
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    elapsed_ms = round(elapsed.microseconds / 1000, 0)

    return({
        "cloud_coverage": cloudy_pct,
        "elapsed_ms": elapsed_ms,
        "oktas": total_oktas,
        "oktas_weighted": total_weighted_oktas,
        "tiles": tile_results
    })


def main():
    setup_logging()
    setup_datadir()

    model, images_classes = load_model()

    mqtt_rx_client = connect_mqtt(settings.mqtt_rx.host, settings.mqtt_rx.port, settings.mqtt_rx.client_id, settings.mqtt_rx.username, settings.mqtt_rx.password)
    # mqtt_tx_client = None

    def image_received_callback(client, userdata, msg):
        logger.info(f"Received image from `{msg.topic}` topic")
        result = analyze_image(model, images_classes, msg.payload)
        pprint(result)
        exit()
        logger.info(f"Analysis results: {result}")

    subscribe(mqtt_rx_client, settings.mqtt_rx.topic, image_received_callback)

    mqtt_rx_client.loop_forever()

if __name__ == "__main__":
    main()
