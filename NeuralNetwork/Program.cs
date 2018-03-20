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

//            var irisModel = new IrisModel(args[0]);
//            irisModel.Train();

//            var xorModel = new XorModel(args[0]);
//            xorModel.Train();
            
//            var mnistModel = new MnistModel(args[0]);
//            mnistModel.Train();

            var mnistConvModel = new MnistConvModel(args[0]);
            mnistConvModel.Train();
        }
    }
}