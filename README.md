# ML Sky Analysis

## Summary

ML Sky Analysis is a tool that analyzes allsky images captured by indi-allsky and estimates the sky condition (cloud coverage) using a trained keras image classification model.

## History

This project was greatly influenced by the awesome work by [Gord Tulloch](https://www.gordtulloch.com/) on the [mlCloudDetect](https://github.com/gordtulloch/mlCloudDetect) project and parts of the code were borrowed from [Kumar Challa's](https://www.kumarchalla.com/) [fork of the project](https://github.com/chvvkumar/mlCloudDetect) (specifically part of the training code was used).

I decided to create my own fork and rename it given that my goals and needs are fairly different from that of the original project. I intend to use it for observatory safety as well, but I want it to be able to do more as a general weather observation tool, so I thought it easiest to just go my own route with it.

With that said, I want to give a HUGE thanks to both Gord and Kumar for their excellent work and code that was able to be borrowed for this project.

I would normally license under Apache 2.0, but since a little of the code was borrowed from their existing GPLv3 projects, just to be safe, I made sure to retain the original licensing.

## License

ML Sky Analysis is licensed under the GPL version 3.

Copyright (c) 2024 Steve Cross <flip@foxhollow.cc>

>    This program is free software: you can redistribute it and/or modify
>    it under the terms of the GNU General Public License as published by
>    the Free Software Foundation, either version 3 of the License, or
>    (at your option) any later version.
>
>    This program is distributed in the hope that it will be useful,
>    but WITHOUT ANY WARRANTY; without even the implied warranty of
>    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
>    GNU General Public License for more details.
>
>    You should have received a copy of the GNU General Public License
>    along with this program.  If not, see <https://www.gnu.org/licenses/>.