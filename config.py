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
