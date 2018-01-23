using System;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("You should provide paths to a training data, to a testing data " +
                                  "and to a folder for storing checkpoints");
                return;
            }

            var descriptor = new NetworkDescriptor(args[0], args[1], args[2], NetworkType.Onehot, 189, 3)
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