# =================================================================================
#
#		ml_sky_analysis - https://www.foxhollow.cc/projects/ml_sky_analysis/
#
#	 ML Sky Analysis is a tool that analyzes allsky images captured by indi-allsky
#    and estimates the sky condition (cloud coverage) using a trained keras image
#    classification model.
#
#        Copyright (c) 2024 Steve Cross <flip@foxhollow.cc>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# =================================================================================

from __future__ import annotations

import os
import json
import logging
import math
import numpy as np
import keras

from datetime import datetime
from paho.mqtt import client as mqtt_client
from PIL import Image, ImageOps
from pprint import pprint

from config import settings

VERSION = "0.1.0"
DEBUG = False
OVERRIDE = None
# OVERRIDE = "ccd1_20241223_214213.jpg"

## Don't change this unless trained with a different size
IMAGE_SIZE = (256, 256)

OKTA_WEIGHT_LIGHT_CLOUDS = 0.5
OKTA_WEIGHT_MEDIUM_CLOUDS = 0.75
OKTA_WEIGHT_THICK_CLOUDS = 1.0

logger = logging.getLogger()

expected_allsky_topics = [
    "latest", # binary image data
    "exp_date", # 2024-12-27 08:20:46
    "exposure", # 0.002794
    "gain", # 0
    "bin", # 1
    "temp", # 20.0
    "sunalt", # 7.7
    "moonalt", # 31.3
    "moonphase", # 11.6
    "mooncycle", # 89.0
    "moonmode", # False
    "night", # False
    "sqm", # 2844443.5
    "stars", # 0
    "latitude", # 30.877
    "longitude", # -84.54
    "elevation", # 40
    "kpindex", # 0.67
    "ovation_max", # 0
    "smoke_rating", # Clear
    "aircraft", # 0
    "sidereal_time", # 14:08:41.25
]

allsky_obj = {}

def make_allsky_object():
    new_allsky_obj = {}

    for allsky_prop in expected_allsky_topics:
        new_allsky_obj[allsky_prop] = None

    return new_allsky_obj

def create_mqtt_client(client_id: string, username: string, password: string):
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

    return client

def connect_mqtt_client(client: mqtt_client, host: string, port: number, ):
    logger.info(f"Connecting to MQTT broker at {host}:{port}")
    client.connect(host, port)

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

    logger.info(f"ML Sky Analysis v{VERSION} starting...")

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

    original_file = os.path.join(settings.datadir, "original.jpg")
    cropped_file = os.path.join(settings.datadir, "cropped.jpg")
    
    if OVERRIDE is not None:
        original_file = os.path.join(settings.datadir, OVERRIDE)
    else:
        # write the original file received from mqtt
        with open(original_file, "wb") as f:
            f.write(imageBytes)

    # load the original file we just wrote. this could be done all in memory without
    # first saving the original file, but i like having the original to see for testing
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
    total_oktas = 0
    total_weighted_oktas = 0
    tile_mapping = []
    # cloudy_count = 0

    # crop the tiles
    for y in range (0, crop_each_direction):
        for x in range(0, crop_each_direction):
            x_offset = x * tile_w
            y_offset = y * tile_h

            tiles.append(cropped_image.crop((x_offset, y_offset, x_offset+tile_w, y_offset+tile_h)))

    # create an array to use for inference that will contain all of the tiles
    data = np.ndarray(shape=(total_tiles,) + IMAGE_SIZE + (3,), dtype=np.float32)
    data_index = 0

    # loop through the tiles and perform inference on each
    for i, tile in enumerate(tiles):
        # skip discarded tiles
        if i in settings.discard_crop_sections:
            continue

        # this is used for later determining which tile was predicted
        tile_mapping.append(i)

        # resize save the cropped tile
        tile_path = os.path.join(settings.datadir, f"tile_{i}.jpg")
        image = tile.resize(IMAGE_SIZE)
        image.save(tile_path)

        # add the tile to the analysis array
        data[data_index] = np.array(image) / 255.0
        data_index += 1


    predictions = model.predict(data, verbose=0)

    if DEBUG:
        pprint(predictions)

    for i, prediction in enumerate(predictions):
        tile_num = tile_mapping[i]

        top_score_index = np.argmax(prediction)
        image_class = images_classes[top_score_index]
        confidence_score = float(prediction[top_score_index])

        ## calculate the weighted oktas value using prediction confidence for
        ## each relevant class
        clear_night_confidence = float(prediction[0])
        light_clouds_night_confidence = float(prediction[1])
        medium_clouds_night_confidence = float(prediction[2])
        thick_clouds_night_confidence = float(prediction[3])

        total_weighted_oktas += light_clouds_night_confidence * OKTA_WEIGHT_LIGHT_CLOUDS * okta_factor
        total_weighted_oktas += medium_clouds_night_confidence * OKTA_WEIGHT_MEDIUM_CLOUDS * okta_factor
        total_weighted_oktas += thick_clouds_night_confidence * OKTA_WEIGHT_THICK_CLOUDS * okta_factor

        if DEBUG:
            pprint(prediction, depth=4)
            logger.debug(f"Tile {tile_num:02} result: {image_class.ljust(6)} / {round(confidence_score * 100, 2)}% confidence")

        okta_score = 0

        if 'light_clouds' in image_class:
            okta_score = OKTA_WEIGHT_LIGHT_CLOUDS
        elif 'medium_clouds' in image_class:
            okta_score = OKTA_WEIGHT_MEDIUM_CLOUDS
        elif 'thick_clouds' in image_class:
            okta_score = OKTA_WEIGHT_THICK_CLOUDS

        total_oktas += (okta_score * okta_factor)

        tile_results.append({
            "tile": tile_num,
            "class": image_class,
            "confidence": round(confidence_score, 3)
        })


    total_oktas = int(round(total_oktas, 0))
    
    # in theory, this sanity check shouldn't be needed.. but leaving it anyways since an okta
    # can never be negative nor can it be over 8
    if total_oktas < 0:
        total_oktas = 0
    elif total_oktas > 8:
        total_oktas = 8

    return({
        # calculate the cloud coverage based on the weighted okta value
        "cloud_coverage": round((total_weighted_oktas / 8), 3),
        "elapsed_ms": int(round((datetime.now() - start_time).microseconds / 1000, 0)),
        "oktas": total_oktas,
        "oktas_weighted": int(round(total_weighted_oktas, 0)),
        "tiles": tile_results
    })

def check_allsky_receive_complete(allsky_obj):
    for allsky_prop in expected_allsky_topics:
        if allsky_obj[allsky_prop] is None:
            return False

    return True

def image_received_callback(client, userdata, msg):
    allsky_topic = str(msg.topic).removeprefix(f"{settings.mqtt.allsky_topic}/")

    if allsky_topic in expected_allsky_topics:
        if allsky_topic == "latest": 
            logger.info(f"Received image from `{msg.topic}` topic")
            userdata["allsky_obj"][allsky_topic] = msg.payload
        else:
            payload = msg.payload.decode("utf-8")
            logger.info(f"Received `{payload}` from `{msg.topic}` topic")
            userdata["allsky_obj"][allsky_topic] = payload

        done = check_allsky_receive_complete(userdata["allsky_obj"])

        logger.debug(f"Allsky object done: {done}")

        if done:
            # process
            analysis_result = analyze_image(userdata["model"], userdata["images_classes"], userdata["allsky_obj"]["latest"])

            if DEBUG:
                pprint(analysis_result)
            
            logger.info(f"Analysis results: {analysis_result}")

            ## send mqtt message
            send_object = {
                "date": userdata["allsky_obj"]["exp_date"],
                "camera_info": {
                    "exposure": float(userdata["allsky_obj"]["exposure"]), # 0.002794
                    "gain": int(userdata["allsky_obj"]["gain"]), # 0
                    "bin": int(userdata["allsky_obj"]["bin"]), # 1
                    "temp": float(userdata["allsky_obj"]["temp"]), # 20.0
                },
                "astro": {
                    "sunalt": float(userdata["allsky_obj"]["sunalt"]), # 7.7
                    "moonalt": float(userdata["allsky_obj"]["moonalt"]), # 31.3
                    "moonphase": float(userdata["allsky_obj"]["moonphase"]), # 11.6
                    "mooncycle": float(userdata["allsky_obj"]["mooncycle"]), # 89.0
                    "moonmode": userdata["allsky_obj"]["moonmode"] == 'True', # False
                    "night": userdata["allsky_obj"]["night"] == 'True', # False
                    "sidereal_time": userdata["allsky_obj"]["sidereal_time"], # 14:08:41.25
                },
                "location": {
                    "latitude": float(userdata["allsky_obj"]["latitude"]), # 30.877
                    "longitude": float(userdata["allsky_obj"]["longitude"]), # -84.54
                    "elevation": int(userdata["allsky_obj"]["elevation"]), # 40
                },
                "sky": {
                    "cloud_coverage": analysis_result["cloud_coverage"],
                    "oktas": analysis_result["oktas"],
                    "oktas_weighted": analysis_result["oktas_weighted"],
                    "sqm": float(userdata["allsky_obj"]["sqm"]), # 2844443.5
                    "stars": int(userdata["allsky_obj"]["stars"]), # 0
                    "kpindex": float(userdata["allsky_obj"]["kpindex"]), # 0.67
                    "ovation_max": float(userdata["allsky_obj"]["ovation_max"]), # 0
                    "smoke_rating": userdata["allsky_obj"]["smoke_rating"], # Clear
                    "aircraft": int(userdata["allsky_obj"]["aircraft"]), # 0

                },
                "analysis": analysis_result,
            }

            if DEBUG:
                pprint(send_object)

            result = client.publish(settings.mqtt.tx_topic, json.dumps(send_object), retain=True)

            status = result[0]
            if status == 0:
                logger.info(f"Sent sky analysis to topic `{settings.mqtt.tx_topic}`")
            else:
                logger.error(f"Failed to send sky analysis to topic {settings.mqtt.tx_topic}")
            
            ## reset the allsky object
            userdata["allsky_obj"] = make_allsky_object()


def main():
    # global allsky_obj

    setup_logging()
    setup_datadir()

    model, images_classes = load_model()

    mqtt_client = create_mqtt_client(settings.mqtt.client_id, settings.mqtt.username, settings.mqtt.password)
    mqtt_client.user_data_set({
        "model": model,
        "images_classes": images_classes, 
        "allsky_obj": make_allsky_object()
    })
    connect_mqtt_client(mqtt_client, settings.mqtt.host, settings.mqtt.port)

    # subscribe to latest image binary data topic
    subscribe(mqtt_client, f"{settings.mqtt.allsky_topic}/#", image_received_callback)

    mqtt_client.loop_forever()

if __name__ == "__main__":
    main()
