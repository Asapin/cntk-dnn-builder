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
            var descriptor = new NetworkDescriptor(IrisTrainDataset, IrisTestDataset, _checkpointPath, NetworkType.Onehot, 4, 3)
            {
                BatchSize = 10,
                EpochCheckpoint = 1,
                Epochs = 350,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f
            };

            ILayer[] layers =
            {
                new SimpleLayer(5, Activation.Tanh),
                new SimpleLayer(4, Activation.Tanh),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}