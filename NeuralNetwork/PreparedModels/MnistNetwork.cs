using System;
using System.Collections.Generic;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;
using NeuralNetwork.Utils;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for classifying MNIST dataset.
    /// The network automatically downloads MNIST train and text datasets, which are about 130 MB in size.
    /// Current configuration is capable of achieving ~98.8% accuracy on test data in ~15 epochs.
    /// </summary>
    public class MnistNetwork
    {
        private const string MnistTrainDataset = "../../Datasets/mnist/train.csv";
        private const string MnistTestDataset = "../../Datasets/mnist/test.csv";

        private readonly string _checkpointPath;

        public MnistNetwork(string checkpointPath)
        {
            _checkpointPath = checkpointPath;
        }

        public void Train()
        {
            Console.WriteLine("Preparing MNIST dataset...");
            DownloadUtils.DownloadMnist();
            Console.WriteLine("Preparing MNIST dataset finished.");
            var descriptor = new NetworkDescriptor(MnistTrainDataset, MnistTestDataset, _checkpointPath, NetworkType.Onehot, 28 * 28, 10)
            {
                BatchSize = 256,
                EvaluateFrequency = 1,
                CheckpointFrequency = 100,
                Epochs = 50,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f,
                DynamicLearningRate = new List<DynamicRate>
                {
                    new DynamicRate(1, 1f),
                    new DynamicRate(1, 0.003f),
                    new DynamicRate(1, 0.001f)
                },
                EpochSize = 5
            };

            ILayer[] layers =
            {
                new SimpleLayer(2000, Activation.LeakyReLU),
                new SimpleLayer(1500, Activation.LeakyReLU),
                new SimpleLayer(1000, Activation.Tanh, true)
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}