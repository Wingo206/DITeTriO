using System.Data.Common;
using System.Runtime.InteropServices;
using TetrEnvironment;
using TetrEnvironment.Constants;
using TetrLoader;
using TetrLoader.Enum;
using Environment = TetrEnvironment.Environment;
using static TetrEnvironment.Constants.Tetromino;
using System.Text;

string rawFilepath = "../../data/raw_replays/";
string procsessedFilepath = "../../data/processed_replays/";

string filename = "test2";

string filepath = rawFilepath + filename + ".ttrm";

// header of the csv
StringBuilder header = new StringBuilder();
for (int j = 0; j < 200; j++)
{
    header.Append("b_"+j+",");
}
header.Append("curr_piece,curr_piece_x,curr_piece_y,curr_piece_rot,");
header.Append("hold_piece,can_hold,");
header.Append("queue_0,queue_1,queue_2,queue_3,queue_4,");
header.Append("last_moveleft,last_moveright,last_softdrop,last_rotate_cw,last_rotate_ccw,last_rotate_180,last_harddrop,last_hold,");
header.Append("moveleft,moveright,softdrop,rotate_cw,rotate_ccw,rotate_180,harddrop,hold");

string headerString = header.ToString();
using (StreamReader reader = new StreamReader(filepath))
{
	string content = reader.ReadToEnd();
	//parse json to IReplayData
	var replayData =
		ReplayLoader.ParseReplay(ref content, Util.IsMulti(ref content) ? ReplayKind.TTRM : ReplayKind.TTR);
    
	Replay replay = new Replay(replayData);

    // for each replay in the file
    for (int i = 0; i < replayData.GetGamesCount(); i++)
    {
        Console.WriteLine("On replay " + (i+1) + " / " + replayData.GetGamesCount());

        // set up the output file
        string outFilepath = procsessedFilepath + filename + "_" + i + ".csv";
        using (StreamWriter writer = new StreamWriter(outFilepath))
        {
            // write the header
            writer.WriteLine(headerString);

            // load the replay
            replay.LoadGame(i);
            bool[] lastInputs = new bool[8];
            int[] framesSinceChange = new int[8];
            while (true)
            {
                LogBoard(replay.Environments, writer);
                LogInputs(replay.Environments, writer, lastInputs, framesSinceChange);

                if (!replay.NextFrame())
                {
                    break;
                }
            }
        }
        // stream writer ends when the block ends

        Console.WriteLine("Wrote file for game " + i);

    }

}

void LogBoard(List<Environment> environments, StreamWriter writer)
{
    // for (int playerIndex = 0; playerIndex < environments.Count; playerIndex++)
    // for now, just log the first player
    for (int playerIndex = 0; playerIndex < 1; playerIndex++)
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
        writer.Write(row.ToString());
    }
}

void LogInputs(List<Environment> environments, StreamWriter writer, bool[] lastInputs, int[] framesSinceChange)
{
    for (int playerIndex = 0; playerIndex < 1; playerIndex++)
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
            if (env.PressingKeys[i] != lastInputs[i])
            {
                // this key has changed
                framesSinceChange[i] = 0;
            } else {
                // key is the same
                framesSinceChange[i]++;
            }

            // add frames since change to the string
            framesSinceChangeString += framesSinceChange[i] + ",";
        }

        // done with the row
        writer.WriteLine(framesSinceChangeString + curInputsString);
    }
}

// change from minotype to 0-6
char encodeMinotype(MinoType mino)
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