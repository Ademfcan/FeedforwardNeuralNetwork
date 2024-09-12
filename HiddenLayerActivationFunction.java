import java.util.Random;

public enum HiddenLayerActivationFunction {
    RELU {
        @Override
        public double Function(double x) {
            return HiddenLayerActivationFunction.relu(x);
        }

        @Override
        public double Derivative(double x) {
            return HiddenLayerActivationFunction.reluDerivative(x);
        }

        @Override
        double initWeight(int nodesIn, int nodesOut) {
            return HiddenLayerActivationFunction.heWeightInit(nodesIn);
        }
    },
    SIGMOID {
        @Override
        public double Function(double x) {
            return HiddenLayerActivationFunction.sigmoid(x);
        }

        @Override
        public double Derivative(double x) {
            return HiddenLayerActivationFunction.sigmoidDerivative(x);
        }

        @Override
        double initWeight(int nodesIn, int nodesOut) {
            return HiddenLayerActivationFunction.xavierWeightInit(nodesIn, nodesOut);
        }
    },
    TANH {
        @Override
        public double Function(double x) {
            return HiddenLayerActivationFunction.tanH(x);
        }

        @Override
        public double Derivative(double x) {
            return HiddenLayerActivationFunction.tanHDerivative(x);
        }

        @Override
        double initWeight(int nodesIn, int nodesOut) {
            return HiddenLayerActivationFunction.xavierWeightInit(nodesIn, nodesOut);

        }
    },
    LEAKYRELU {
        @Override
        public double Function(double x) {
            return HiddenLayerActivationFunction.leakyRelu(x);
        }

        @Override
        public double Derivative(double x) {
            return HiddenLayerActivationFunction.leakyReluDerivative(x);
        }

        @Override
        double initWeight(int nodesIn, int nodesOut) {
            return HiddenLayerActivationFunction.heWeightInit(nodesIn);
        }
    },
    SILU {
        @Override
        public double Function(double x) {
            return HiddenLayerActivationFunction.silu(x);
        }

        @Override
        public double Derivative(double x) {
            return HiddenLayerActivationFunction.siluDerivative(x);
        }

        @Override
        double initWeight(int nodesIn, int nodesOut) {
            return HiddenLayerActivationFunction.xavierWeightInit(nodesIn, nodesOut);
        }
    },
    SWISH {
        @Override
        public double Function(double x) {
            return x * HiddenLayerActivationFunction.swish(x);
        }

        @Override
        public double Derivative(double x) {
            return swishDerivative(x);
        }

        @Override
        double initWeight(int nodesIn, int nodesOut) {
            return HiddenLayerActivationFunction.heWeightInit(nodesIn);
        }
    };

    abstract double Function(double x);

    abstract double Derivative(double x);

    abstract double initWeight(int nodesIn, int nodesOut);


    private static double relu(double x) {
        return Math.max(0, x);
    }


    private static double reluDerivative(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    private static double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    private static double silu(double x) {
        if (x > 0) {
            return sigmoid(x);
        } else {
            return 0;
        }
    }

    private static double siluDerivative(double x) {
        if (x > 0) {
            return sigmoidDerivative(x);
        } else {
            return 0;
        }
    }

    private static double tanH(double x) {
        return Math.tanh(x);
    }

    private static double tanHDerivative(double x) {
        double tanX = Math.tanh(x);
        return 1 - (tanX * tanX);
    }

    private static final double leakyCoeff = .03f;

    private static double leakyRelu(double x) {
        if (x > 0) {
            return x;
        } else {
            return x * leakyCoeff;
        }
    }

    private static double leakyReluDerivative(double x) {
        if (x > 0) {
            return 1;
        } else {
            return leakyCoeff;
        }
    }

    private static double swish(double x) {
        return x * sigmoid(x);
    }

    private static double swishDerivative(double x) {
        return sigmoid(x) + x * sigmoidDerivative(x);
    }

    private static final Random weightInitializer = new Random();

    private static double xavierWeightInit(double neuronsIn, double neuronsOut) {
        return weightInitializer.nextGaussian() * Math.sqrt(2 / (neuronsIn + neuronsOut));

    }

    private static double heWeightInit(double neuronsIn) {
        return weightInitializer.nextGaussian() * Math.sqrt(2 / neuronsIn);

    }


}
