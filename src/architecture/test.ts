// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';

// Define the U-Net model
function createUNet(inputShape) {
  const inputs = tf.input({ shape: inputShape });

  // Encoder (Downsampling)
  let conv1 = tf.layers
    .conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' })
    .apply(inputs);
  conv1 = tf.layers
    .conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' })
    .apply(conv1);
  let pool1 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    .apply(conv1);

  let conv2 = tf.layers
    .conv2d({
      filters: 128,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(pool1);
  conv2 = tf.layers
    .conv2d({
      filters: 128,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(conv2);
  let pool2 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    .apply(conv2);

  let conv3 = tf.layers
    .conv2d({
      filters: 256,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(pool2);
  conv3 = tf.layers
    .conv2d({
      filters: 256,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(conv3);
  let pool3 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    .apply(conv3);

  let conv4 = tf.layers
    .conv2d({
      filters: 512,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(pool3);
  conv4 = tf.layers
    .conv2d({
      filters: 512,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(conv4);
  let drop4 = tf.layers.dropout({ rate: 0.5 }).apply(conv4);
  let pool4 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    .apply(drop4);

  // Middle layer
  let conv5 = tf.layers
    .conv2d({
      filters: 1024,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(pool4);
  conv5 = tf.layers
    .conv2d({
      filters: 1024,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(conv5);
  let drop5 = tf.layers.dropout({ rate: 0.5 }).apply(conv5);

  // Decoder (Upsampling)
  let up6 = tf.layers
    .conv2dTranspose({
      filters: 512,
      kernelSize: 2,
      strides: [2, 2],
      padding: 'same',
    })
    .apply(drop5);
  up6 = tf.layers
    .concatenate({ axis: -1 })
    .apply([up6 as SymbolicTensor, conv4 as SymbolicTensor]);
  up6 = tf.layers
    .conv2d({
      filters: 512,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(up6);
  up6 = tf.layers
    .conv2d({
      filters: 512,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(up6);

  let up7 = tf.layers
    .conv2dTranspose({
      filters: 256,
      kernelSize: 2,
      strides: [2, 2],
      padding: 'same',
    })
    .apply(up6);
  up7 = tf.layers
    .concatenate({ axis: -1 })
    .apply([up7 as SymbolicTensor, conv3 as SymbolicTensor]);
  up7 = tf.layers
    .conv2d({
      filters: 256,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(up7);
  up7 = tf.layers
    .conv2d({
      filters: 256,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(up7);

  let up8 = tf.layers
    .conv2dTranspose({
      filters: 128,
      kernelSize: 2,
      strides: [2, 2],
      padding: 'same',
    })
    .apply(up7);
  up8 = tf.layers
    .concatenate({ axis: -1 })
    .apply([up8 as SymbolicTensor, conv2 as SymbolicTensor]);
  up8 = tf.layers
    .conv2d({
      filters: 128,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(up8);
  up8 = tf.layers
    .conv2d({
      filters: 128,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    })
    .apply(up8);

  let up9 = tf.layers
    .conv2dTranspose({
      filters: 64,
      kernelSize: 2,
      strides: [2, 2],
      padding: 'same',
    })
    .apply(up8);
  up9 = tf.layers
    .concatenate({ axis: -1 })
    .apply([up9 as SymbolicTensor, conv1 as SymbolicTensor]);
  up9 = tf.layers
    .conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' })
    .apply(up9);
  up9 = tf.layers
    .conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' })
    .apply(up9);

  // Final output layer
  let outputs = tf.layers
    .conv2d({
      filters: 1,
      kernelSize: 1,
      padding: 'same',
      activation: 'sigmoid',
    })
    .apply(up9);

  // Define the model
  const model = tf.model({
    inputs: inputs,
    outputs: outputs as SymbolicTensor,
  });

  return model;
}

// Example usage
const model = createUNet([256, 256, 1]);
console.log(model);
