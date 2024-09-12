public enum CostFunctions {
    MSE,
    CROSSENTROPY;


    public static double getMSECOST(double[] output, int expectedIndex) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            double out = output[i];
            if (i == expectedIndex) {
                // expected will have a probablity of 1
                sum += Math.pow(out - 1, 2);
            } else {
                // else the expected will be zero
                sum += Math.pow(out, 2);
            }
        }
        return sum / (2 * output.length);
    }

    public static double[] getMSEGradients(double[] output, int expectedIndex) {
        double[] gradients = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            gradients[i] = -1*(1.0 / output.length) * (output[i] - (i == expectedIndex ? 1 : 0));
        }
        return gradients;
    }

    public static double getCROSSENTROPYLOSS(double[] output, int expectedIndex) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            int y = i == expectedIndex ? 1 : 0;
            double out = output[i];
            sum += y * Math.log(out) + (1 - y) * Math.log(1 - out);
        }
        return sum / (2 * output.length);
    }


}
