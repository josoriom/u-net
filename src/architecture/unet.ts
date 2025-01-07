import {
  LayersModel,
  layers,
  SymbolicTensor,
  model,
  serialization,
} from '@tensorflow/tfjs-node';

import { decoderBlock } from './utilities/decoderBlock.js';
import { encoderBlock } from './utilities/encoderBlock.js';
import { ResizeSymbolicTensor } from './utilities/ResizeSymbolicTensor.js';

serialization.registerClass(ResizeSymbolicTensor);

export function unetModel(
  inputShape: [number, number, number],
  numberOfClasses = 1,
): LayersModel {
  const inputs = layers.input({ shape: inputShape });
  const e1 = encoderBlock(inputs, 64);
  const e2 = encoderBlock(e1, 128);
  const e3 = encoderBlock(e2, 256);
  const e4 = encoderBlock(e3, 512);
  let b1 = layers
    .conv2d({
      filters: 1024,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
    .apply(e4) as SymbolicTensor;
  b1 = layers.activation({ activation: 'relu' }).apply(b1) as SymbolicTensor;
  b1 = layers
    .conv2d({
      filters: 1024,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
    .apply(b1) as SymbolicTensor;

  b1 = layers.batchNormalization().apply(b1) as SymbolicTensor;
  b1 = layers.activation({ activation: 'relu' }).apply(b1) as SymbolicTensor;

  console.log({ b1: b1.shape, e4: e4.shape });

  const d1 = decoderBlock(b1, e4, 512);
  const d2 = decoderBlock(d1, e3, 256);
  const d3 = decoderBlock(d2, e2, 128);
  const d4 = decoderBlock(d3, e1, 64);
  const outputs = layers
    .conv2d({
      filters: numberOfClasses,
      kernelSize: 1,
      padding: 'same',
      activation: 'sigmoid',
    })
    .apply(d4) as SymbolicTensor;

  const result = model({ inputs, outputs, name: 'U-Net' });
  return result;
}
