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
    /// Current configuration is capable of achieving ~98.8% accuracy on test data.
    /// </summary>
    public class MnistModel : AbstractModel
    {
        private const string MnistTrainDataset = "../../Datasets/mnist/train.csv";
        private const string MnistTestDataset = "../../Datasets/mnist/test.csv";

        public MnistModel(string checkpointPath) : base(checkpointPath)
        {
        }

        protected override void Before()
        {
            Console.WriteLine("Preparing MNIST dataset...");
            DownloadUtils.DownloadMnist();
            Console.WriteLine("Preparing MNIST dataset finished.");
        }

        protected override NetworkDescriptor GetNetworkDescriptor()
        {
            return new NetworkDescriptor(MnistTrainDataset, MnistTestDataset, CheckpointPath, new[] { 28 * 28 }, 10)
            {
                BatchSize = 256,
                EvaluateFrequency = 1,
                CheckpointFrequency = 150,
                EpochsToTrain = 100,
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
                TrainingScheduleEpochs = 5
            };
        }

        protected override IEnumerable<ILayer> GetLayers()
        {
            ILayer[] layers =
            {
                new FullyConnectedLayer(Activation.LeakyReLU, 2000),
                new FullyConnectedLayer(Activation.LeakyReLU, 1500),
                new DropoutLayer(0.2), 
                new BatchNormalizationLayer(Activation.Tanh, 1000), 
            };

            return layers;
        }
    }
}