import { Tensor4D } from '@tensorflow/tfjs-node';

export interface TrainingData {
  training: {
    x: Tensor4D;
    y: Tensor4D;
  };
  validation: {
    x: Tensor4D;
    y: Tensor4D;
  };
  batchSize: number;
  trainingSize: number;
  validationSize: number;
}
