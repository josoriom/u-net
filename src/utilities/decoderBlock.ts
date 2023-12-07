import { SymbolicTensor, layers } from '@tensorflow/tfjs-node';

export function decoderBlock(
  inputs: SymbolicTensor,
  skip: SymbolicTensor,
  filters: number,
) {
  let dencoder = layers
    .conv2dTranspose({
      filters,
      kernelSize: 2,
      strides: 2,
      activation: 'relu',
      padding: 'same',
    })
    .apply(inputs) as SymbolicTensor;
  dencoder = layers
    .concatenate({ axis: 3 })
    .apply([dencoder, skip]) as SymbolicTensor;
  dencoder = layers
    .conv2d({
      filters,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
    .apply(dencoder) as SymbolicTensor;
  dencoder = layers.batchNormalization().apply(dencoder) as SymbolicTensor;
  dencoder = layers
    .activation({ activation: 'relu' })
    .apply(dencoder) as SymbolicTensor;
  dencoder = layers
    .conv2d({
      filters,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
    .apply(dencoder) as SymbolicTensor;
  dencoder = layers.batchNormalization().apply(dencoder) as SymbolicTensor;
  dencoder = layers
    .activation({ activation: 'relu' })
    .apply(dencoder) as SymbolicTensor;
  return dencoder;
}
