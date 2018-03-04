using System;
using CNTK;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;
using NeuralNetwork.Utils;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for classifying MNIST dataset using convolutional neural networks.
    /// The network automatically downloads MNIST train and text datasets, which are about 130 MB in size.
    /// Current configuration is capable of achieving ~99.3% accuracy on test data.
    /// </summary>
    public class MnistConvNetwork
    {
        private const string MnistTrainDataset = "../../Datasets/mnist/train.csv";
        private const string MnistTestDataset = "../../Datasets/mnist/test.csv";

        private readonly string _checkpointPath;

        public MnistConvNetwork(string checkpointPath)
        {
            _checkpointPath = checkpointPath;
        }

        public void Train()
        {
            Console.WriteLine("Preparing MNIST dataset...");
            DownloadUtils.DownloadMnist();
            Console.WriteLine("Preparing MNIST dataset finished.");
            var descriptor = new NetworkDescriptor(MnistTrainDataset, MnistTestDataset, _checkpointPath, 
                NetworkType.Onehot, new[] { 28, 28, 1 }, 10)
            {
                BatchSize = 256,
                EvaluateFrequency = 1,
                CheckpointFrequency = 150,
                Epochs = 100,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f,
                EpochSize = 5
            };

            ILayer[] layers =
            {
                new ConvolutionLayer(Activation.ReLU, 5, 5, 6),
                new PoolingLayer(PoolingType.Max, 2, 2, 2, 2),
                new ConvolutionLayer(Activation.ReLU, 3, 3, 16),
                new PoolingLayer(PoolingType.Max, 2, 2, 2, 2),
                new DropoutLayer(0.5), 
                new FullyConnectedLayer(Activation.ReLU, 120),
                new FullyConnectedLayer(Activation.ReLU, 84),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}