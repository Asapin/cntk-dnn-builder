using System;
using NeuralNetwork.PreparedModels;

namespace NeuralNetwork
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("You should provide path to a folder for storing checkpoints");
                return;
            }

//            var irisNetwork = new IrisNetwork(args[0]);
//            irisNetwork.Train();

//            var xorNetwork = new XorNetwork(args[0]);
//            xorNetwork.Train();
            
//            var mnistNetwork = new MnistNetwork(args[0]);
//            mnistNetwork.Train();

            var mnistConvNetwork = new MnistConvNetwork(args[0]);
            mnistConvNetwork.Train();
        }
    }
}