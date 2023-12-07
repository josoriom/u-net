import { LayersModel, metrics, train } from '@tensorflow/tfjs-node';

import { diceCoefficient } from '../metrics/diceCoefficient.js';
import { TrainingData } from '../types/TrainData.js';

export async function trainModel(
  model: LayersModel,
  data: TrainingData,
  options: {
    epochs?: number;
    learningRate?: number;
  } = {},
) {
  const { epochs = 50, learningRate = 0.001 } = options;
  const { training, validation, batchSize } = data;
  model.compile({
    optimizer: train.adam(learningRate),
    loss: metrics.binaryCrossentropy,
    metrics: [diceCoefficient],
  });

  await model.fit(training.x, training.y, {
    validationData: [validation.x, validation.y],
    shuffle: false,
    batchSize,
    epochs,
  });
}
