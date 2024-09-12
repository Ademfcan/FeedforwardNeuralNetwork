public enum OutputLayerActivationFunctions {
    SOFTMAXX,
    CROSSENTROPYLOSS;


    public static double[] softMaxx(double[] outPutLayer) {
        double divisor = getDivisor(outPutLayer);
        double[] out = new double[outPutLayer.length];
        for (int i = 0; i < outPutLayer.length; i++) {
            out[i] = softMaxx(outPutLayer[i], divisor);
        }
        return out;

    }

    private static double softMaxx(double out, double divisor) {
        return Math.exp(out) / divisor;


    }

    private static double getDivisor(double[] outPutLayer) {
        double divisor = 0;
        for (double d : outPutLayer) {
            divisor += Math.exp(d);
        }
        return divisor;
    }

    public static double[] softMaxDerivatives(double[] nonSoftMaxxed, double[] softMaxxed) {
        double[] out = new double[softMaxxed.length];
        double divisorSoft = getDivisor(softMaxxed);
        double divisorNonSoft = getDivisor(nonSoftMaxxed);
        for (int i = 0; i < nonSoftMaxxed.length; i++) {
            double deriv = 0;

            for (int j = 0; j < softMaxxed.length; j++) {
                deriv += softMaxx(softMaxxed[j], divisorSoft) * (i == j ? 1 - softMaxx(nonSoftMaxxed[i], divisorNonSoft): -1*softMaxx(softMaxxed[j],divisorSoft));
            }
            out[i] = deriv;

        }
        return out;
    }
}
