namespace DITeTriOPreProcessor;
using Environment = TetrEnvironment.Environment;
using static TetrEnvironment.Constants.Tetromino;
using System.Text;

public static class ProcessorUtils
{

    public static void LogBoard(List<Environment> environments, StreamWriter[] writers)
    {
        // for (int playerIndex = 0; playerIndex < environments.Count; playerIndex++)
        // for now, just log the first player
        for (int playerIndex = 0; playerIndex < environments.Count; playerIndex++)
        {
            StringBuilder row = new StringBuilder();
            Environment env = environments[playerIndex];

            // board blocks
            MinoType[] board = env.GameData.Board;
            for (int y = 20; y < 40; y++)
            {
                for (int x = 0; x < 10; x++)
                {
                    row.Append(encodeMinotype(board[x + y * 10]));
                    row.Append(',');
                }
            }

            // current piece id, x, y, rotation
            row.Append(encodeMinotype(env.GameData.Falling.Type) + ",");
            row.Append(env.GameData.Falling.X + ",");
            row.Append((int)Math.Ceiling(env.GameData.Falling.Y) + ",");
            row.Append(env.GameData.Falling.R + ",");

            // hold piece id and canhold
            row.Append(encodeMinotype(env.GameData.Hold) + ",");
            row.Append(!env.GameData.HoldLocked + ",");

            // next queue piece ids
            List<MinoType> visibleQueue = env.GameData.Bag.Take(5).ToList();
            foreach (MinoType m in visibleQueue) 
            {
                row.Append(encodeMinotype(m) + ",");
            }

            // write to the csv
            writers[playerIndex].Write(row.ToString());
        }
    }

    public static void LogInputs(List<Environment> environments, StreamWriter[] writers, bool[,] lastInputs, int[,] framesSinceChange)
    {
        for (int playerIndex = 0; playerIndex < environments.Count; playerIndex++)
        {
            Environment env = environments[playerIndex];

            // inputs - loop through and update values
            string curInputsString = "";
            string framesSinceChangeString = "";
            for (int i = 0; i < 8; i++) 
            {
                curInputsString += env.PressingKeys[i] ? 1:0;
                if (i != 7) 
                {
                    curInputsString += ",";
                }

                // update the frames since change
                if (env.PressingKeys[i] != lastInputs[playerIndex, i])
                {
                    // this key has changed
                    framesSinceChange[playerIndex, i] = 0;
                } else {
                    // key is the same
                    // if pressed, decrease (go negative). If unpressed, increase (go positive)
                    if (env.PressingKeys[i])
                    {
                        framesSinceChange[playerIndex, i]--;
                    }
                    else 
                    {
                        framesSinceChange[playerIndex, i]++;
                    }
                }

                // add frames since change to the string
                framesSinceChangeString += framesSinceChange[playerIndex, i] + ",";
            }

            // done with the row
            writers[playerIndex].WriteLine(framesSinceChangeString + curInputsString);
        }
    }

    // change from minotype to 0-6
    public static char encodeMinotype(MinoType mino)
    {
        switch (mino)
        {
            case MinoType.Empty:
                return '0';
            case MinoType.I:
                return '1';
            case MinoType.J:
                return '2';
            case MinoType.L:
                return '3';
            case MinoType.O:
                return '4';
            case MinoType.S:
                return '5';
            case MinoType.T:
                return '6';
            case MinoType.Z:
                return '7';
            case MinoType.Garbage:
                return '8';
            
            default:
                return '0';
        }
    }

}