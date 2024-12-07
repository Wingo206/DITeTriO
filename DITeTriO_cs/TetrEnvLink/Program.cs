using Environment = TetrEnvironment.Environment;
using TetrEnvironment;
using TetrLoader;
using TetrLoader.Enum;
using static TetrEnvironment.Constants.Tetromino;
using TetrLoader.JsonClass.Event;
using System.Reflection;
using System.Net;
using static DITeTriOPreProcessor.ProcessorUtils;


// goal: load a replay to initialize the environment, remove the events, replace with new events

class Program
{

    static void Main(string[] args)
    {
        // listen to the pipe for instructions from python
        string cmdPipeName = args[0] + "/cmdPipe";
        string envPipeName = args[0] + "/envPipe";

        Console.WriteLine($"Connecting to the pipe at {cmdPipeName}");
        using (FileStream cmdPipeStream = new FileStream(cmdPipeName, FileMode.Open, FileAccess.Read))
        using (StreamReader cmdReader = new StreamReader(cmdPipeStream))
        {
            Console.WriteLine($"Connecting to the pipe at {envPipeName}");
            using (FileStream envPipeStream = new FileStream(envPipeName, FileMode.Open, FileAccess.Write))
            using (StreamWriter envWriter = new StreamWriter(envPipeStream))
            {
                Console.WriteLine("Connected to both pipes!");
                envWriter.AutoFlush = true;

                RunGame(cmdReader, envWriter);
                // string cmd = cmdReader.ReadLine();
                // Console.WriteLine(cmd);
                // string information = "Hello from C#";

                // envWriter.WriteLine(information);
                // envWriter.WriteLine("hello");
                // envWriter.WriteLine(); // termination
            }
        }
    }

    static void RunGame(StreamReader cmdReader, StreamWriter envWriter)
    {
        string replayFilepath = cmdReader.ReadLine();
        using (StreamReader reader = new StreamReader(replayFilepath))
        {
            string content = reader.ReadToEnd();
            //parse json to IReplayData
            var replayData =
                ReplayLoader.ParseReplay(ref content, Util.IsMulti(ref content) ? ReplayKind.TTRM : ReplayKind.TTR);
            
            Replay replay = new Replay(replayData);
            replay.LoadGame(3);
            replay.JumpFrame(250);

            Console.WriteLine(replay);

            Environment env = replay.Environments[0];
            Console.WriteLine(env.FrameInfo.CurrentFrame);
            Console.WriteLine(env.FrameInfo._currentIndex);
            Console.WriteLine(env.TotalFrame);

            Console.WriteLine(replay.Environments[1].FrameInfo._currentIndex);
            printBoard(env);

            // hijack the _events readonly list extremely grossly
            FakeReadOnlyList<Event> newEvents = new FakeReadOnlyList<Event>();
            Type envType = env.GetType();
            FieldInfo fieldInfo = envType.GetField("_events", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!;

            // look at the existing events to figure out how they work
            List<Event> oldEvents = (List<Event>)fieldInfo.GetValue(env);

            // replace with bogus list 
            fieldInfo.SetValue(env, newEvents);

            for (int k = 0; k < env.FrameInfo._currentIndex; k++)
            {
                newEvents.Add(oldEvents[k]);
                Console.WriteLine(k);
            }

            // start events
            /* newEvents.Add(oldEvents[0]); */
            /* newEvents.Add(oldEvents[1]); */
            /* newEvents.Add(oldEvents[2]); */

            // start simulation loop
            List<Environment> envs = [env];
            StreamWriter[] writers = [envWriter];
            bool[,] lastInputs = new bool[1, 8];
            int[,] framesSinceChange = new int[1, 8];
            while(true)
            {
                // write to the pipe
                LogBoard(envs, writers);
                LogInputs(envs, writers, lastInputs, framesSinceChange);
                for (int j = 0; j < lastInputs.GetLength(1); j++)
                {
                    lastInputs[0, j] = env.PressingKeys[j];
                }

                // read the next command
                Console.WriteLine("Waiting for command");
                string cmd = cmdReader.ReadLine();
                if (cmd is null)
                {
                    break;
                }
                Console.WriteLine(cmd);

                // process the command and create events
                bool[] nextFrameInputs = cmd.Split(' ').Select(x => x == "1").ToArray();
                for (int j = 0; j < lastInputs.GetLength(1); j++)
                {
                    if (lastInputs[0, j] != nextFrameInputs[j])
                    {
                        // this input has changed, create an event for next frame
                        EventKeyInputData downData = new EventKeyInputData();
                        downData.key = (KeyType)j;
                        downData.subframe = 0.1;
                        downData.hoisted = false;
                        EventKeyInput inputEvent;
                        // iterate and if different, create an event
                        if (lastInputs[0,j] && !nextFrameInputs[j])
                        {
                            // we want to release this input
                            inputEvent = new EventKeyInput(null, env.FrameInfo.CurrentFrame, EventType.Keyup, downData);
                        } 
                        else
                        {
                            // we want to press this input
                            inputEvent = new EventKeyInput(null, env.FrameInfo.CurrentFrame, EventType.Keydown, downData);
                        }
                        newEvents.Add(inputEvent);
                    }
                }


                // next frame
                env.NextFrame();

                // testing
                printBoard(env);

            }


        }

    }

    static void printBoard(Environment env)
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
}
            // // add some new events
            // for (int i = 0; i < 5; i++) {

            //     EventKeyInputData downData = new EventKeyInputData();
            //     downData.key = KeyType.HardDrop;
            //     downData.subframe = 0;
            //     downData.hoisted = false;

            //     EventKeyInput downEvent = new EventKeyInput(null, i*2, EventType.Keydown, downData);
            //     newEvents.Add(downEvent);
            //     EventKeyInput upEvent = new EventKeyInput(null, i*2+1, EventType.Keyup, downData);
            //     newEvents.Add(upEvent);
            // }

            // foreach (Event e in oldEvents)
            // {
            //     Console.Write(e.id + ", " + e.frame + ", " + e.type);
            //     if (e is EventKeyInput)
            //     {
            //         EventKeyInput e2 = (EventKeyInput) e;
            //         EventKeyInputData data = e2.data;
            //         Console.Write("," + data.hoisted + ", " + data.key + ", " + data.subframe);
            //     }
            //     else
            //     {
            //         Console.Write(e.ToString());
            //     }
            //     // Console.WriteLine();
            // }

            // bool pressed = false;
            // for (int i = 0; i < 10; i++) {
            //     Queue<Event> nextEvents = new Queue<Event>();

            //     // add an event for the next frame
            //     EventKeyInputData downData = new EventKeyInputData();
            //     downData.key = KeyType.HardDrop;
            //     downData.subframe = 0.1;
            //     downData.hoisted = false;
            //     if (!pressed) 
            //     {
            //         EventKeyInput downEvent = new EventKeyInput(null, env.FrameInfo.CurrentFrame + 1, EventType.Keydown, downData);
            //         // nextEvents.Enqueue(downEvent);
            //         newEvents.Add(downEvent);
            //         pressed = true;
            //     }
            //     else
            //     {
            //         EventKeyInput upEvent = new EventKeyInput(null, env.FrameInfo.CurrentFrame + 1, EventType.Keyup, downData);
            //         // nextEvents.Enqueue(upEvent);
            //         newEvents.Add(upEvent);
            //         pressed = false;
            //     }
            //     // 
            //     // env.NextFrame(nextEvents);
            //     env.NextFrame();
            //     Console.WriteLine(env.TotalFrame);
            //     Console.WriteLine("Current Frame: " + env.FrameInfo.CurrentFrame);
            //     Console.WriteLine("Current Index: " + env.FrameInfo._currentIndex);
            //     printBoard(env);
            // }
