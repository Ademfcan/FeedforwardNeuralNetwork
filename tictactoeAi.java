import jdk.jfr.MemoryAddress;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class tictactoeAi extends FeedforwardNetwork{

    public tictactoeAi(double globalLearningrate,HiddenLayer... hiddenLayers) {
        super(75, globalLearningrate, new InputLayer(10), new OutputLayer(9, 9, OutputLayerActivationFunctions.SOFTMAXX), hiddenLayers);
    }
    public void trainAi(List<String> boards, List<Integer> correctMoveIndexes, boolean isDebug,boolean isGradientCheck) {
        if (boards.size() != correctMoveIndexes.size()) {
            System.out.println("Boards and CorrectMoveIndexes must be same length!");
            return;
        }

        double[][] output = new double[boards.size()][boardLength];
        for (int i = 0; i < boards.size(); i++) {
            output[i] = turnBoardToInput(boards.get(i));
        }
        super.train(output, correctMoveIndexes.stream().mapToInt(Integer::intValue).toArray(),isDebug,isGradientCheck);


    }

    private final int boardLength = 10;
    private final int inputLength = 9;

    private double[] turnBoardToInput(String board) {
        // string will be comma separated with the x's and o's in a flattened 1x9 array
        // eg: "X,X,_,O,_,_,X,O,O"
        // an x will have a value of -1
        // an O will have a value of 1
        double[] input = new double[boardLength];
        String[] split = board.split(",");

        int piecediff = 0;
        for (String s : split) {
            if (s.equalsIgnoreCase("x")) {
                piecediff++;
            } else if (s.equalsIgnoreCase("o")) {
                piecediff--;
            }
        }
        // this is complety reliant on the fact that usually x goes first
        boolean isXTurn = piecediff <= 0;

        if (split.length != inputLength) {
            System.out.println("Incorrect input provided! : " + board);
            System.out.println("Must be a string with x's and o's separated by commas");
            System.out.println("Mark empty spaces with any other string");
            System.out.println("Must have 9 spots");
            return null;
        }
        for (int i = 0; i < split.length; i++) {
            int val = 0;
            String lower = split[i].toLowerCase();
            lower = lower.trim();
            if(lower.equals("o") || split[i].equals("0") || split[i].equals("O")){
                val = 1;
            }
            else if(lower.equals("x")){
                val = -1;

            }
            input[i] = val;
        }
        input[input.length - 1] = isXTurn ? 1 : -1;

        return input;
    }


    public String getMove(String board) {
        double[] outPut = super.forwardPass(turnBoardToInput(board));
        System.out.println("Ai output:\n" + Arrays.toString(outPut));

        int moveIndx = super.getHighestProbIndex(outPut);
        int piecediff = 0;
        for (String s : board.split(",")) {
            if (s.equalsIgnoreCase("x")) {
                piecediff++;
            } else if (s.equalsIgnoreCase("o")) {
                piecediff--;
            }
        }
        // this is complety reliant on the fact that usually x goes first
        boolean isXTurn = piecediff <= 0;
        // 0,0 origin is located in the top left corner
        int xColumn = moveIndx % 3;
        int yColumn = moveIndx / 3;
        return (isXTurn ? "X" : "O") + "to X: " + xColumn + " Y: " + yColumn + "moveIndx: " + moveIndx;


    }


}
