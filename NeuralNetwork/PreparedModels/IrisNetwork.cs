using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    public class IrisNetwork
    {
        public void Train(string checkpointPath)
        {
            const string trainDataPath = "../../datasets/iris/train.csv";
            const string testDataPath = "../../datasets/iris/train.csv";

            var descriptor = new NetworkDescriptor(trainDataPath, testDataPath, checkpointPath, NetworkType.Onehot, 4, 3)
            {
                BatchSize = 10,
                EpochCheckpoint = 1,
                Epochs = 2000,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f
            };

            ILayer[] layers =
            {
                new SimpleLayer(10, Activation.LeakyReLU),
                new SimpleLayer(20, Activation.LeakyReLU),
                new SimpleLayer(10, Activation.LeakyReLU),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}