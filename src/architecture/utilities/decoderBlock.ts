import { SymbolicTensor, layers } from '@tensorflow/tfjs-node';
import { ResizeSymbolicTensor } from './ResizeSymbolicTensor';

export function decoderBlock(
  inputs: SymbolicTensor,
  skip: SymbolicTensor,
  filters: number,
) {
  let decoder = layers
    .conv2dTranspose({
      filters,
      kernelSize: 2,
      strides: 2,
      activation: 'relu',
      padding: 'same',
    })
    .apply(inputs) as SymbolicTensor;
  const resizedSkip = new ResizeSymbolicTensor(
    decoder.shape[1],
    decoder.shape[2],
  ).apply(skip) as SymbolicTensor;
  decoder = layers
    .concatenate({ axis: 3 })
    .apply([decoder, resizedSkip]) as SymbolicTensor;
  decoder = layers
    .conv2d({ filters, kernelSize: 3, activation: 'relu', padding: 'same' })
    .apply(decoder) as SymbolicTensor;
  decoder = layers.batchNormalization().apply(decoder) as SymbolicTensor;
  decoder = layers
    .activation({ activation: 'relu' })
    .apply(decoder) as SymbolicTensor;
  decoder = layers
    .conv2d({ filters, kernelSize: 3, activation: 'relu', padding: 'same' })
    .apply(decoder) as SymbolicTensor;
  decoder = layers.batchNormalization().apply(decoder) as SymbolicTensor;
  decoder = layers
    .activation({ activation: 'relu' })
    .apply(decoder) as SymbolicTensor;
  return decoder;
}
