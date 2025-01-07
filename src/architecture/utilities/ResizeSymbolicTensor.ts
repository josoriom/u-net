import * as tf from '@tensorflow/tfjs';
import { serialization, Tensor, Tensor4D, Shape } from '@tensorflow/tfjs';
import { layers } from '@tensorflow/tfjs-layers';

/**
 * A custom layer that applies tf.image.resizeBilinear to SymbolicTensors
 * within the Keras-style graph.
 */
export class ResizeSymbolicTensor extends layers.Layer {
  static className = 'ResizeBilinear'; // for serialization

  private targetHeight: number;
  private targetWidth: number;

  constructor(targetHeight: number, targetWidth: number) {
    super({});
    this.targetHeight = targetHeight;
    this.targetWidth = targetWidth;
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape {
    let shape: Shape;
    if (Array.isArray(inputShape[0])) {
      // Then inputShape[0] is itself a (number|null)[] => a Shape
      shape = inputShape[0] as Shape;
    } else {
      // inputShape is already a single Shape
      shape = inputShape as Shape;
    }

    return [shape[0], this.targetHeight, this.targetWidth, shape[3]];
  }

  call(
    inputs: Tensor | Tensor[],
    kwargs: { [key: string]: any },
  ): Tensor | Tensor[] {
    return tf.tidy(() => {
      const input = (Array.isArray(inputs) ? inputs[0] : inputs) as Tensor4D;
      // Perform the symbolic resize op:
      // NOTE: This works on SymbolicTensors because we're inside a Layer subclass.
      return tf.image.resizeBilinear(input, [
        this.targetHeight,
        this.targetWidth,
      ]);
    });
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return {
      ...baseConfig,
      targetHeight: this.targetHeight,
      targetWidth: this.targetWidth,
    };
  }

  getClassName() {
    return ResizeSymbolicTensor.className;
  }

  static fromConfig<T>(
    // @ts-expect-error I know
    cls: serialization.SerializableConstructor<T>,
    config: {},
  ) {
    const { targetHeight, targetWidth } = config as any;
    return new cls(targetHeight, targetWidth);
  }
}
serialization.SerializationMap.register(ResizeSymbolicTensor);
