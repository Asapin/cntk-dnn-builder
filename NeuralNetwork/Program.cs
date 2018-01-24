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

            var irisNetwork = new IrisNetwork(args[0]);
            irisNetwork.Train();
        }
    }
}