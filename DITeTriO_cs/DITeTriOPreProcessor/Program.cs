using System.Data.Common;
using System.Runtime.InteropServices;
using TetrEnvironment;
using TetrEnvironment.Constants;
using TetrLoader;
using TetrLoader.Enum;
using Environment = TetrEnvironment.Environment;
using Tetromino.MinoType;

Console.WriteLine("Hello, World!");

string rawFilepath = "../../data/raw_replays/";
string procsessedFilepath = "../../data/procssed_replays/";

string filename = "test2"

string filepath = rawFilepath + filename + ".ttrm";
using (StreamReader reader = new StreamReader(filepath))
{
	string content = reader.ReadToEnd();
	//parse json to IReplayData
	var replayData =
		ReplayLoader.ParseReplay(ref content, Util.IsMulti(ref content) ? ReplayKind.TTRM : ReplayKind.TTR);
    
	Replay replay = new Replay(replayData);

    while (true)
    {
        // for each replay in the file
        for (int i = 0; i < replayData.GetGamesCount(); i++)
        {
            Console.WriteLine("On replay " + i + " / " + replayData.GetGamesCount());

            // set up the output file
            string outFilepath = procsessedFilepath + filename + "_" + i + ".ttrm";
            using (StreamWriter writer = new StreamWriter(outFilepath))
            {
                //TODO
                writer.writeLine("header")

                // load the replay
                replay.LoadGame(i);
                while (true)
                {
                    LogBoard(replay.environments[0], writer);
                    if (!replay.NextFrame())
                    {
                        break;
                    }
                }
            }
            // stream writer ends when the block ends

            Console.writeLine("Wrote file for game " + i);

        }
    }

}

void LogBoard(List<Environment> environments, StreamWriter writer)
{
    // for (int playerIndex = 0; playerIndex < environments.Count; playerIndex++)
    // for now, just log the first player
    for (int playerIndex = 0; playerIndex < 1; playerIndex++)
    {
        // board blocks (200)

        // current piece id, x, y, rotation

        // hold piece id and canhold

        asdf
        // next queue piece ids

        // frames since input change

        // inputs

        
    }

}

// change from minotype to 0-6
char encodeMinotype(Tetromino.MinoType mino)
{
    switch (mino)
    {

    }
}