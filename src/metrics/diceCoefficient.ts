import { Tensor, sum, scalar } from '@tensorflow/tfjs-node-node';

export function diceCoefficient(yTrue: Tensor, yPred: Tensor): Tensor {
  const intersection = sum(yTrue.mul(yPred)).cast('float32');
  const union = sum(yTrue.add(yPred)).cast('float32');
  const dice = intersection.mul(scalar(2.0)).div(union.add(intersection));
  return dice;
}
