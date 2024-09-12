import java.util.Arrays;

public class tictactoePos {

    char[] pos;

    int correctMoveIndex;

    public tictactoePos(char[] pos, int correctMoveIndex) {
        this.pos = pos;
        this.correctMoveIndex = correctMoveIndex;
    }
    public String getPosWithoutBrackets() {
        String brackets = Arrays.toString(pos);
        return brackets.substring(1,brackets.length()-1);
    }




}
