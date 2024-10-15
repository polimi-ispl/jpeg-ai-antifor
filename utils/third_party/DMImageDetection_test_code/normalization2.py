'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''


from PIL import Image
import numbers


def padding_wrap(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    new_img = Image.new(img.mode, output_size)
    for x_offset in range(0, output_size[0], img.size[0]):
        for y_offset in range(0, output_size[1], img.size[1]):
            new_img.paste(img, (x_offset, y_offset))

    return new_img


class PaddingWarp():
    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        return padding_wrap(img, self.siz)

