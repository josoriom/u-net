import {
  LayersModel,
  layers,
  SymbolicTensor,
  model,
} from '@tensorflow/tfjs-node';

import { decoderBlock } from './utilities/decoderBlock.js';
import { encoderBlock } from './utilities/encoderBlock.js';

export function unetModel(
  inputShape: [number, number, number],
  numberOfClasses = 1,
): LayersModel {
  const inputs = layers.input({ shape: inputShape });
  const { encoder: e1, skip: skip1 } = encoderBlock(inputs, 64);
  const { encoder: e2, skip: skip2 } = encoderBlock(e1, 128);
  const { encoder: e3, skip: skip3 } = encoderBlock(e2, 256);
  const { encoder: e4, skip: skip4 } = encoderBlock(e3, 512);
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
  const d1 = decoderBlock(b1, skip4, 512);
  const d2 = decoderBlock(d1, skip3, 256);
  const d3 = decoderBlock(d2, skip2, 128);
  const d4 = decoderBlock(d3, skip1, 64);
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
