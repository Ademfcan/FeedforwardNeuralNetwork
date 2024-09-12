import java.util.Random;

public abstract class Layer {


    public double[] neuronValues;

    public int getNeuronCount() {
        return neuronCount;
    }

    public final int neuronCount;

    public Layer(int neuronCount) {
        this.neuronValues = new double[neuronCount];
        this.neuronCount = neuronCount;

    }


}
