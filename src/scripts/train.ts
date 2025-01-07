import { fileCollectionFromPath } from 'filelist-utils';

import { unetModel } from '../architecture/unet.js';

import { getTrainData } from './getTrainData.js';
import { trainModel } from './trainModel.js';

const size = 256;
const epochs = 2;
const learningRate = 0.001;

const model = unetModel([size, size, 3], 1);
const dataFolder = new URL('../../data/train/', import.meta.url);
const x = await fileCollectionFromPath(new URL('x', dataFolder).pathname);
const y = await fileCollectionFromPath(new URL('y', dataFolder).pathname);
const data = await getTrainData(x, y);

await trainModel(model, { ...data, batchSize: 32 }, { epochs, learningRate });

// eslint-disable-next-line no-console
console.log(`:::::::::: Saving the model :::::::::::`);
await model.save(
  `file://${new URL(`./model_${epochs}`, import.meta.url).pathname}`,
);
data.training.x.dispose();
data.training.y.dispose();
data.validation.x.dispose();
data.validation.y.dispose();
model.dispose();
