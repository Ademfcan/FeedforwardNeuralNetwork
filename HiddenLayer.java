import java.util.Arrays;
import java.util.Random;

public class HiddenLayer extends Layer {
    public double[] getNeuronValues() {
        return neuronValues;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutPutSize() {
        return outPutSize;
    }

    protected double[][] layerWeights;

    protected double[] layerBiases;
    private final HiddenLayerActivationFunction function;

    protected final int inputSize;
    protected final int outPutSize;


    protected double learningRate;

    protected double[] previousLayerNeurons;

    public HiddenLayer(int inputSize, int outPutSize, HiddenLayerActivationFunction function) {
        super(outPutSize);
        layerWeights = new double[outPutSize][inputSize];
        layerBiases = new double[outPutSize];

        this.function = function;
        this.inputSize = inputSize;
        this.outPutSize = outPutSize;
        initWeights(layerWeights, inputSize, outPutSize);
        initBiases(layerBiases, inputSize, outPutSize);
    }

    public void initLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void initPreviousLayerNeurons(double[] previousLayerNeurons) {
        this.previousLayerNeurons = previousLayerNeurons;
    }


    public void ForwardPropagate() {
        if (previousLayerNeurons.length != inputSize) {
            System.out.println("Error! previous layer size does not match the input size of this layer");
            System.out.println("Previous layer size: " + previousLayerNeurons.length + " input size: " + inputSize);
        } else {
            for (int i = 0; i < outPutSize; i++) {
                double weightedSum = 0d;
                for (int j = 0; j < inputSize; j++) {
                    weightedSum += layerWeights[i][j] * previousLayerNeurons[j];
                }
                weightedSum += layerBiases[i];
                double activated = function.Function(weightedSum);

                super.neuronValues[i] = activated;
            }
        }
    }

    public void adjustWeights(double[] Gradients) {
        for (int i = 0; i < outPutSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                layerWeights[i][j] -= learningRate * Gradients[i] ;//* previousLayerNeurons[j];
            }
        }

    }

    public void adjustBiases(double[] gradients) {
        for (int i = 0; i < neuronCount; i++) {
            // Update bias using gradient descent
            layerBiases[i] -= learningRate * gradients[i];
        }
    }


    public double[] calculateLayerGradients(double[] GradientsAbove, double[][] layerWeightsAbove) {
        double[] gradients = new double[neuronCount]; // Gradients for each neuron in this layer
        for (int i = 0; i < neuronCount; i++) {
            double weightedSum = 0d;
            for (int j = 0; j < inputSize; j++) {
                weightedSum += layerWeights[i][j] * previousLayerNeurons[j];
            }
            double activationDerivative = function.Derivative(weightedSum + layerBiases[i]);
            double gradientSum = 0d;
            for (int j = 0; j < GradientsAbove.length; j++) { // Loop over neurons in the next layer
                gradientSum += GradientsAbove[j] * layerWeightsAbove[j][i];
            }
            gradients[i] = activationDerivative * gradientSum;
        }
        return gradients;
    }


    private void initWeights(double[][] weights, int inputSize, int outputSize) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = function.initWeight(inputSize, outputSize);
            }
        }
    }

    private void initBiases(double[] weights, int inputSize, int outputSize) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = function.initWeight(inputSize, outputSize);

        }
    }


}
