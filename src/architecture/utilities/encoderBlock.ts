import { SymbolicTensor, layers } from '@tensorflow/tfjs-node';

export function encoderBlock(inputs: SymbolicTensor, filters: number) {
  let encoder = layers
    .conv2d({
      filters,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
    .apply(inputs) as SymbolicTensor;
  encoder = layers.batchNormalization().apply(encoder) as SymbolicTensor;
  encoder = layers
    .activation({ activation: 'relu' })
    .apply(encoder) as SymbolicTensor;
  encoder = layers
    .conv2d({
      filters,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
    .apply(encoder) as SymbolicTensor;
  encoder = layers.batchNormalization().apply(encoder) as SymbolicTensor;
  encoder = layers
    .activation({ activation: 'relu' })
    .apply(encoder) as SymbolicTensor;
  encoder = layers
    .maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
    })
    .apply(encoder) as SymbolicTensor;
  return encoder;
}
