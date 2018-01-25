using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for solving XOR problem.
    /// Current configuration is capable of achieving 100% accuracy on test data in ~10 epochs
    /// </summary>
    public class XorNetwork
    {
        private const string XorTrainDataset = "../../Datasets/xor/train.csv";
        private const string XorTestDataset = "../../Datasets/xor/test.csv";

        private readonly string _checkpointPath;

        public XorNetwork(string checkpointPath)
        {
            _checkpointPath = checkpointPath;
        }
        
        public void Train()
        {
            var descriptor = new NetworkDescriptor(XorTrainDataset, XorTestDataset, _checkpointPath, NetworkType.Onehot, 2, 2)
            {
                BatchSize = 4,
                EpochCheckpoint = 1,
                Epochs = 130,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.125f
            };

            ILayer[] layers =
            {
                new SimpleLayer(8, Activation.ReLU),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}