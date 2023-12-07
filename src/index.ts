import { unetModel } from './unet.js';

const size = 256;
const inputShape: [number, number, number] = [size, size, 3];
const testModel = unetModel(inputShape, 2);
testModel.summary();
