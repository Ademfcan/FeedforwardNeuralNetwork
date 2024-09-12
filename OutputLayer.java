import java.util.Arrays;
import java.util.Objects;

public class OutputLayer extends HiddenLayer {

    OutputLayerActivationFunctions function;

    double[] nonNormalizedNeuronValues;

    public OutputLayer(int inputSize, int outPutSize, OutputLayerActivationFunctions function) {
        super(inputSize, outPutSize, HiddenLayerActivationFunction.SIGMOID);
        this.function = function;
    }

    @Override
    public void ForwardPropagate() {
        if (previousLayerNeurons.length != getInputSize()) {
            System.out.println("Error! previous layer size does not match the input size of this layer");
            System.out.println("Previous layer size: " + previousLayerNeurons.length + " input size: " + inputSize);
        } else {
            for (int i = 0; i < getOutPutSize(); i++) {
                double weightedSum = 0d;
                for (int j = 0; j < getInputSize(); j++) {
                    weightedSum += layerWeights[i][j] * previousLayerNeurons[j];
                }
                super.neuronValues[i] = weightedSum + layerBiases[i];
            }
            nonNormalizedNeuronValues = super.neuronValues;
            switch (function) {
                case SOFTMAXX -> {
                    super.neuronValues = OutputLayerActivationFunctions.softMaxx(super.neuronValues);
                }
            }
        }

    }

    public double[] calculateCostFunctionGradient(double[] output, int expectedIndex) {
        // Compute gradients of MSE loss with respect to softmax output
        double[] gradients = CostFunctions.getMSEGradients(output, expectedIndex);
        return gradients;
    }

    public double[] calculateGradientsOutput(double[] gradients) {
        double[] derivatives = null;
        if (Objects.requireNonNull(function) == OutputLayerActivationFunctions.SOFTMAXX) {
            derivatives = OutputLayerActivationFunctions.softMaxDerivatives(nonNormalizedNeuronValues, neuronValues);
//            System.out.println("Softmaxx derivatives: " + Arrays.toString(derivatives));
            // Multiply softmax derivatives by gradients of MSE loss with respect to softmax output
            for (int i = 0; i < outPutSize; i++) {
                derivatives[i] *= gradients[i];
            }
        }
        return derivatives; // Return the computed derivatives
    }


}
