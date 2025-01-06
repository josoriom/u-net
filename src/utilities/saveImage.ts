import { writeFileSync } from 'fs';

import { Tensor } from '@tensorflow/tfjs-node-node';
import { encode, Image, ImageColorModel } from 'image-js';

export function saveImage(tensor: Tensor, kind: string, size) {
  const array = tensor.arraySync()[0];
  const result = new Image(size, size, {
    colorModel: ImageColorModel.GREY,
  }).fill(255);

  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      result.setPixel(j, i, [array[i][j][0] * 255]);
    }
  }
  writeFileSync(new URL(`./${kind}.png`, import.meta.url), encode(result));
}
