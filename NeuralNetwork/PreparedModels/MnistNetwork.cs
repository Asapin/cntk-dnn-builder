using System;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;
using NeuralNetwork.Utils;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for classifying MNIST dataset.
    /// The network automatically downloads MNIST train and text datasets, which are about 130 MB in size.
    /// Current configuration is capable of achieving ~98.5% accuracy on test data in ~15 epochs.
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
                BatchSize = 200,
                EvaluateFrequency = 1,
                Epochs = 50,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f
            };

            ILayer[] layers =
            {
                new SimpleLayer(2000, Activation.LeakyReLU),
                new SimpleLayer(1500, Activation.LeakyReLU),
                new SimpleLayer(1000, Activation.Tanh),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}