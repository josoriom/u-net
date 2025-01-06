/* eslint-disable no-await-in-loop */
import { tensor4d } from '@tensorflow/tfjs-node';
import { FileCollection } from 'filelist-utils';
import { ImageColorModel, decode } from 'image-js';

import { TrainingData } from '../types/TrainData';

import { getIndices } from './getIndices.js';

export async function getTrainData(
  x: FileCollection,
  y: FileCollection,
  options: { width?: number; height?: number; channels?: number } = {},
): Promise<TrainingData> {
  const { width = 256, height = 256, channels = 3 } = options;
  const xSize = width * height * channels;
  const ySize = width * height * 1;

  const itemsLength = x.files.length;
  const datasets = getIndices(itemsLength);
  const images = datasets.map((item) => new Float32Array(item.length * xSize));
  const labels = datasets.map((item) => new Float32Array(item.length * ySize));

  for (let i = 0; i < itemsLength; i++) {
    const name = x.files[i].name;
    const xCollectionItem = x.files.find((item) => item.name === name);
    const [dataset, index] = getPosition(datasets, i);
    const xBuffer = new Uint8Array(await xCollectionItem.arrayBuffer());
    const xImage = decode(xBuffer).resize({ width, height });
    const rgb =
      xImage.colorModel !== ImageColorModel.RGB
        ? xImage.convertColor(ImageColorModel.RGB)
        : xImage;
    const xData = rgb.getRawImage().data;
    for (let j = 0; j < xData.length; j++) {
      images[dataset][index * xSize + j] = xData[j] / 255;
    }
    const yCollectionItem = y.files.find((item) => item.name === name);
    const yBuffer = new Uint8Array(await yCollectionItem.arrayBuffer());
    const yImage = decode(yBuffer).resize({ width, height });
    const grey =
      yImage.colorModel !== ImageColorModel.GREY
        ? yImage.convertColor(ImageColorModel.GREY)
        : yImage;
    const yData = grey.getRawImage().data;
    for (let k = 0; k < yData.length; k++) {
      labels[dataset][index * ySize + k] = yData[k] / 255;
    }
  }
  const trainingSize = datasets[0].length;
  const validationSize = datasets[1].length;
  return {
    training: {
      x: tensor4d(images[0], [trainingSize, width, height, channels]),
      y: tensor4d(labels[0], [trainingSize, width, height, 1]),
    },
    validation: {
      x: tensor4d(images[1], [validationSize, width, height, channels]),
      y: tensor4d(labels[1], [validationSize, width, height, 1]),
    },
    batchSize: itemsLength,
    trainingSize,
    validationSize,
  };
}

function getPosition(arr: number[][], target: number) {
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr[i].length; j++) {
      if (arr[i][j] === target) {
        return [i, j];
      }
    }
  }
  return [-1, -1];
}
