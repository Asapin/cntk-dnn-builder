using System;
using System.IO;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("You should provide path to a folder with train and test data");
                return;
            }

            var trainPath = Path.Combine(args[0], "train.csv");
            var testPath = Path.Combine(args[0], "test.csv");
            var checkpointPath = args[0];

            var descriptor = new NetworkDescriptor(trainPath, testPath, checkpointPath, NetworkType.Onehot, 189, 3)
            {
                BatchSize = 100,
                EpochCheckpoint = 10,
                Epochs = 5000,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f
            };

            ILayer[] layers =
            {
                new DropoutLayer(0.3), 
                new SimpleLayer(200, Activation.LeakyReLU),
                new SimpleLayer(100, Activation.LeakyReLU),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}