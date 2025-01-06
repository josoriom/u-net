import { readFileSync } from 'fs';

import { Tensor, loadLayersModel, tensor4d } from '@tensorflow/tfjs-node';
import { decode } from 'image-js';

import { saveImage } from './utilities/saveImage';

const modelPath = new URL('./new_256_50/model.json', import.meta.url);
const size = 256;
const model = await loadLayersModel(modelPath.href);
const image = decode(
  new Uint8Array(
    readFileSync(new URL(`../data/train/x/0.png`, import.meta.url)),
  ),
);

const flatImage = image.getRawImage().data;
const imageBatch = new Float32Array(size * size * 3);
for (let j = 0; j < flatImage.length; j++) {
  imageBatch[j] = flatImage[j] / 255;
}
const imageTensor = tensor4d(imageBatch, [1, size, size, 3]);
const prediction = model.predict(imageTensor) as Tensor;
imageTensor.dispose();
model.dispose();
saveImage(prediction, 'test', size);
