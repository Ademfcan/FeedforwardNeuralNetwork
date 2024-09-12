import java.util.Arrays;

public class FeedforwardNetwork {
    private final InputLayer inputLayer;

    private final OutputLayer outputLayer;

    private final HiddenLayer[] hiddenLayers;

    private final int batchSize;
    private final double globalLearningRate;

    private final int DEFAULTBATCHSIZE = 8;

    private final double DEFAULTLEARNINGRATE = .001d;

    private final int maxLayerSize;

    private final double minCost;

    private final int maxIters = 30000;

    private final double maxErrorJump = .005d;

    public FeedforwardNetwork(InputLayer inputLayer, OutputLayer outputLayer, HiddenLayer... hiddenLayers) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayers = hiddenLayers;
        this.batchSize = DEFAULTBATCHSIZE;
        this.globalLearningRate = DEFAULTLEARNINGRATE;
        for (int i = 0; i < hiddenLayers.length; i++) {
            double[] previousLayerNeurons;
            if (i == 0) {
                previousLayerNeurons = inputLayer.neuronValues;
            } else {
                previousLayerNeurons = hiddenLayers[i - 1].neuronValues;
            }
            hiddenLayers[i].initLearningRate(globalLearningRate);
            hiddenLayers[i].initPreviousLayerNeurons(previousLayerNeurons);
        }
        outputLayer.initLearningRate(globalLearningRate);
        outputLayer.initPreviousLayerNeurons(hiddenLayers[hiddenLayers.length - 1].neuronValues);
        this.maxLayerSize = getMaxLayerSize(hiddenLayers);
        this.minCost = .005f;
    }

    public FeedforwardNetwork(int batchSize, double globalLearningRate, InputLayer inputLayer, OutputLayer outputLayer, HiddenLayer... hiddenLayers) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayers = hiddenLayers;
        this.batchSize = batchSize;
        this.globalLearningRate = globalLearningRate;
        for (int i = 0; i < hiddenLayers.length; i++) {
            double[] previousLayerNeurons;
            if (i == 0) {
                previousLayerNeurons = inputLayer.neuronValues;
            } else {
                previousLayerNeurons = hiddenLayers[i - 1].neuronValues;
            }
            hiddenLayers[i].initLearningRate(globalLearningRate);
            hiddenLayers[i].initPreviousLayerNeurons(previousLayerNeurons);
        }
        outputLayer.initLearningRate(globalLearningRate);
        outputLayer.initPreviousLayerNeurons(hiddenLayers[hiddenLayers.length - 1].neuronValues);
        this.maxLayerSize = getMaxLayerSize(hiddenLayers);
        this.minCost = .005f;


    }

    public  void train(double[][] inputs, int[] expectedIndexes,boolean isDebug,boolean isGradientCheck) {
        if (inputs.length != expectedIndexes.length) {
            System.out.println("inputs and expectedIndexes must be same length!");
            return;
        }
        if (inputs.length < batchSize) {
            System.out.println("Too few training inputs for batch size!");
            System.out.println("Try changing batch size: current =" + batchSize);
            return;
        }
        int numBatches = inputs.length / batchSize;
        int lastBatchSize = inputs.length % batchSize;
        if(isDebug){
            double[] networkOut = forwardPass(inputs[inputs.length / 2]);
            double cost = CostFunctions.getMSECOST(networkOut, expectedIndexes[inputs.length / 2]);
            System.out.println("Initial cost: " + cost);
        }
        // do the full batches
        int iterCount = 0;
        double lastCost = 1;
        while (true) {
            for (int i = 0; i < numBatches; i++) {
                trainBatch(inputs, expectedIndexes, batchSize, i, 0,isDebug,isGradientCheck);
            }
            // do a forward inference pass for each input then calculate the error
            // sum upp all errors for the whole batch then average out error;
            // using this average error backpropagate and repeat
            // do whatever remaining batch is left
            if (lastBatchSize != 0) {
                trainBatch(inputs, expectedIndexes, lastBatchSize, 0, numBatches * batchSize,isDebug,isGradientCheck);

            }
            double avgCost = getAverageCost(inputs,expectedIndexes);
            iterCount++;
            if(isDebug){
                System.out.println("#(" + iterCount + ")Average cost = " + avgCost);
            }
            if(avgCost < minCost || iterCount > maxIters || (avgCost-lastCost > maxErrorJump) || Double.isNaN(avgCost)){
                break;
            }


        }
        if(isDebug){
            double[] networkOutF = forwardPass(inputs[inputs.length / 2]);
            double costF = CostFunctions.getMSECOST(networkOutF, expectedIndexes[inputs.length / 2]);
            System.out.println("Final cost: " + costF);
        }

    }

    private double getAverageCost(double[][] inputs,int[] expectedIndexes){
        double totalCost = 0;
        for(int i  = 0;i<inputs.length;i++){
            double[] networkOut = forwardPass(inputs[i]);
            double cost = CostFunctions.getMSECOST(networkOut, expectedIndexes[i]);
            totalCost += cost;
        }
        return totalCost/inputs.length;

    }

    public int getHighestProbIndex(double[] softMaxxedOutput) {
        int highestIndex = -1;
        double highestProb = -1;
        for (int i = 0; i < softMaxxedOutput.length; i++) {
            if (softMaxxedOutput[i] > highestProb) {
                highestIndex = i;
                highestProb = softMaxxedOutput[i];
            }
        }
        return highestIndex;
    }

    private void trainBatch(double[][] batchInputs, int[] batchExpectedIndexes, int batchLength, int callCount, int isEndOfBatchIndex,boolean isDebug,boolean isGradientCheck) {

        double[] outputGradientsAvg = null;
        double[][] hiddenGradientsAvg = new double[hiddenLayers.length][];
        int index;
        if (isEndOfBatchIndex != 0) {
            index = isEndOfBatchIndex;
        } else {
            index = batchLength * callCount;

        }
        for (int i = index; i < index + batchLength; i++) {
            // calculate gradients for each input
            int expectedIndex = batchExpectedIndexes[i];
            double[] batchInput = batchInputs[i];
            double[] networkOut = forwardPass(batchInput);

            double[] costgradients = outputLayer.calculateCostFunctionGradient(networkOut, expectedIndex);
            if(isGradientCheck){
                System.out.println("Network Out: " + Arrays.toString(networkOut));
                System.out.println("Cost gradient: " + Arrays.toString(costgradients));
                System.out.println("Board: " + Arrays.toString(batchInput));
                System.out.println("Expected index: " + expectedIndex);
            }
            double[] outputGradients = outputLayer.calculateGradientsOutput(costgradients);

            double[][] hiddenGradients = new double[hiddenLayers.length][];
            double[][] lastWeights = outputLayer.layerWeights;
            double[] lastGradients = outputGradients;
            for (int j = hiddenLayers.length - 1; j >= 0; j--) {
                // reverse order for backprop
                hiddenGradients[j] = hiddenLayers[j].calculateLayerGradients(lastGradients, lastWeights);
                lastGradients = hiddenGradients[j];
                lastWeights = hiddenLayers[j].layerWeights;

                if (hiddenGradientsAvg[j] == null) {
                    hiddenGradientsAvg[j] = hiddenGradients[j];
                } else {
                    hiddenGradientsAvg[j] = mergeGradients(hiddenGradientsAvg[j], hiddenGradients[j]);
                }

            }
            if (outputGradientsAvg == null) {
                outputGradientsAvg = outputGradients;
            } else {
                outputGradientsAvg = mergeGradients(outputGradientsAvg, outputGradients);
            }

        }
        // average out gradients

        averageGradients(outputGradientsAvg, batchLength);

        for (int i = 0; i < hiddenGradientsAvg.length; i++) {
            hiddenGradientsAvg[i] = averageGradients(hiddenGradientsAvg[i], batchLength);
        }
        outputLayer.adjustWeights(outputGradientsAvg);
        outputLayer.adjustBiases(outputGradientsAvg);
        if(isGradientCheck){
            System.out.println("Output gradients: " + Arrays.toString(outputGradientsAvg));
            Arrays.stream(hiddenGradientsAvg).forEach(d -> System.out.println("Hidden gradient: "  + Arrays.toString(d)));
        }

        for (int i = 0; i < hiddenGradientsAvg.length; i++) {
            hiddenLayers[i].adjustWeights(hiddenGradientsAvg[i]);
            hiddenLayers[i].adjustBiases(hiddenGradientsAvg[i]);
        }



    }

    private void updateLayerLearningRate(double LearningRate) {
        outputLayer.learningRate = LearningRate;
        for (HiddenLayer l : hiddenLayers) {
            l.learningRate = LearningRate;
        }
    }

    private double[] mergeGradients(double[] gradients, double[] newGradients) {
        if (gradients.length != newGradients.length) {
            System.out.println("Error! merging gradients that do not match in size!");
            System.out.println("Gradient 1 size " + gradients.length + " gradient 2 size " + newGradients.length);
            return null;
        }
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] += newGradients[i];

        }
        return gradients;
    }

    private double[] averageGradients(double[] gradients, int batchLength) {
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= (batchLength);
        }
        return gradients;
    }

    public double[] forwardPass(double[] inputs) {
        inputLayer.setInputs(inputs);
        for (HiddenLayer l : hiddenLayers) {
            l.ForwardPropagate();
        }
        outputLayer.ForwardPropagate();
        return outputLayer.neuronValues;
    }

    private int getMaxLayerSize(HiddenLayer[] layers) {
        int maxSize = 0;
        for (HiddenLayer l : layers) {
            if (l.neuronValues.length > maxSize) {
                maxSize = l.neuronValues.length;
            }
        }
        System.out.println(maxSize);
        return maxSize;
    }


}
