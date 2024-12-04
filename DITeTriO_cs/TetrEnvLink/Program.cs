using Environment = TetrEnvironment.Environment;
using TetrEnvironment;
using TetrLoader;
using TetrLoader.Enum;
using static TetrEnvironment.Constants.Tetromino;
using TetrLoader.JsonClass.Event;
using System.Reflection;
using System.Net;

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

// goal: load a replay to initialize the environment, remove the events, replace with new events



using (StreamReader reader = new StreamReader("/Users/brandon/Documents/GitHub/DITeTriO/data/raw_replays/test2.ttrm"))
{
	string content = reader.ReadToEnd();
	//parse json to IReplayData
	var replayData =
		ReplayLoader.ParseReplay(ref content, Util.IsMulti(ref content) ? ReplayKind.TTRM : ReplayKind.TTR);
    
	Replay replay = new Replay(replayData);
    replay.LoadGame(0);
    Console.WriteLine(replay);

    Environment env = replay.Environments[0];
    printBoard(env);

    // hijack the _events readonly list extremely grossly
    FakeReadOnlyList<Event> newEvents = new FakeReadOnlyList<Event>();
    Type envType = env.GetType();
    FieldInfo fieldInfo = envType.GetField("_events", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!;

    // look at the existing events to figure out how they work
    List<Event> oldEvents = (List<Event>)fieldInfo.GetValue(env);
    foreach (Event e in oldEvents)
    {
        Console.Write(e.id + ", " + e.frame + ", " + e.type);
        if (e is EventKeyInput)
        {
            EventKeyInput e2 = (EventKeyInput) e;
            EventKeyInputData data = e2.data;
            Console.Write("," + data.hoisted + ", " + data.key + ", " + data.subframe);
        }
        else
        {
            Console.Write(e.ToString());
        }
        // Console.WriteLine();
    }

    // replace with bogus list 
    fieldInfo.SetValue(env, newEvents);
    // start events
    newEvents.Add(oldEvents[0]);
    newEvents.Add(oldEvents[1]);
    newEvents.Add(oldEvents[2]);
    for (int i = 0; i < 5; i++) {

        EventKeyInputData downData = new EventKeyInputData();
        downData.key = KeyType.HardDrop;
        downData.subframe = 0;
        downData.hoisted = false;

        EventKeyInput downEvent = new EventKeyInput(null, i*2, EventType.Keydown, downData);
        newEvents.Add(downEvent);
        EventKeyInput upEvent = new EventKeyInput(null, i*2+1, EventType.Keyup, downData);
        newEvents.Add(upEvent);
    }

    env.NextFrame();
    env.NextFrame();
    env.NextFrame();
    env.NextFrame();
    env.NextFrame();
    env.NextFrame();
    printBoard(env);





}

void printBoard(Environment env)
{
    MinoType[] board = env.GameData.Board;
    for (int y = 20; y < 40; y++)
    {
        for (int x = 0; x < 10; x++) 
        {
            Console.Write(encodeMinotype(board[x + y * 10]));
        }
        Console.Write("\n");
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
