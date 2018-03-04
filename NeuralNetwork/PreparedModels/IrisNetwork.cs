using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for learning Iris dataset.
    /// Current configuration is capable of achieving 100% accuracy on test data.
    /// </summary>
    public class IrisNetwork
    {
        private const string IrisTrainDataset = "../../Datasets/iris/train.csv";
        private const string IrisTestDataset = "../../Datasets/iris/test.csv";

        private readonly string _checkpointPath;

        public IrisNetwork(string checkpointPath)
        {
            _checkpointPath = checkpointPath;
        }

        public void Train()
        {
            var descriptor = new NetworkDescriptor(IrisTrainDataset, IrisTestDataset, _checkpointPath, 
                NetworkType.Onehot, new[] { 4 }, 3)
            {
                BatchSize = 10,
                EvaluateFrequency = 1,
                CheckpointFrequency = 150,
                Epochs = 100,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.0015f
            };

            ILayer[] layers =
            {
                new FullyConnectedLayer(Activation.LeakyReLU, 9)
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}