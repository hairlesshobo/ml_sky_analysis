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

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print("gpu:")
get_available_gpus()

print("devices:")
print(device_lib.list_local_devices())
