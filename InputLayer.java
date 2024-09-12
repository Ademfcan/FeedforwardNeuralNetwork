public class InputLayer extends Layer {
    public InputLayer(int neuronCount) {
        super(neuronCount);
    }

    public void setInputs(double[] inputValues) {
        System.arraycopy(inputValues, 0, this.neuronValues, 0, inputValues.length);
    }
}
