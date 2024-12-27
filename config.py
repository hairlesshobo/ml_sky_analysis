import os

from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="MSA",
    settings_files=['settings.yaml', 'secrets.yaml'],
    validators=[
        # Latitude of observer
        Validator(
            "latitude",
            default="40.123456",
            cast=float
        ),

        # Longitude of observer
        Validator(
            "longitude",
            default="-80.123456",
            cast=float
        ),

        Validator(
            "crop_l",
            default=0,
            cast=int
        ),

        Validator(
            "crop_t",
            default=0,
            cast=int
        ),

        Validator(
            "crop_r",
            default=0,
            cast=int
        ),

        Validator(
            "crop_b",
            default=0,
            cast=int
        ),

        Validator(
            "crop_sections",
            default=16,
            cast=int
        ),

        Validator(
            "discard_crop_sections",
            default=[],
            cast=list
        ),

        # Model file to use
        Validator(
            "model_data",
            default="keras_model.h5"
        ),

        # Labels file to use
        Validator(
            "model_labels",
            default="labels.txt"
        ),

        # Altitude that the sun has to be at to be full night
        Validator(
            "day_altitude",
            default="-12"
        ),

        Validator(
            "datadir",
            default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        )
    ]
)
