using System.Data.Common;
using System.Runtime.InteropServices;
using TetrEnvironment;
using TetrEnvironment.Constants;
using TetrLoader;
using TetrLoader.Enum;
using Environment = TetrEnvironment.Environment;
using static TetrEnvironment.Constants.Tetromino;
using System.Text;
using static DITeTriOPreProcessor.ProcessorUtils;

// input: "/raw/path filename /output/path player1 player2"
// /Users/brandon/Documents/GitHub/DITeTriO/data/raw_replays test2 /Users/brandon/Downloads bruh broh
// output: /output/path/player1/filename_p0_r0.csv
string input = Console.ReadLine();

string[] splitInput = input.Split(' ');

string rawDir = splitInput[0];
string filename = splitInput[1];
string processedDir = splitInput[2];
string player0 = splitInput[3];
string player1 = splitInput[4];

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

// process the replay
using (StreamReader reader = new StreamReader(rawDir + "/" + filename + ".ttrm"))
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

        // set up the output files
        string outFilepath0 = processedDir + "/" + player0 + "/" + filename + "_p0_r"+i+".csv";
        string outFilepath1 = processedDir + "/" + player1 + "/" + filename + "_p1_r"+i+".csv";
        StreamWriter[] writers = new StreamWriter[2];
        writers[0] = new StreamWriter(outFilepath0);
        writers[1] = new StreamWriter(outFilepath1);
        try {

            // write the header
            foreach (var writer in writers)
            {
                writer.WriteLine(headerString);
            }

            // load the replay
            replay.LoadGame(i);
            bool[,] lastInputs = new bool[2, 8];
            int[,] framesSinceChange = new int[2, 8];
            while (true)
            {
                LogBoard(replay.Environments, writers);
                LogInputs(replay.Environments, writers, lastInputs, framesSinceChange);
                // copy the lastInputs
                for (int playerIndex = 0; playerIndex < 2; playerIndex++)
                {
                    for (int j = 0; j < lastInputs.GetLength(1); j++)
                    {
                        lastInputs[playerIndex, j] = replay.Environments[playerIndex].PressingKeys[j];
                    }
                }

                if (!replay.NextFrame())
                {
                    break;
                }
            }
        }
        finally
        {
            // close the streamwriters
            foreach (var writer in writers)
            {
                writer?.Dispose();
            }

        }

        Console.WriteLine("Wrote file for game " + i);

    }

}