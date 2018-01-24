using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    public class IrisNetwork
    {
        private const string DatasetsIrisTrainCsv = "../../datasets/iris/train.csv";
        private const string DatasetsIrisTestCsv = "../../datasets/iris/test.csv";

        private readonly string _checkpointPath;

        public IrisNetwork(string checkpointPath)
        {
            _checkpointPath = checkpointPath;
        }

        public void Train()
        {
            var descriptor = new NetworkDescriptor(DatasetsIrisTrainCsv, DatasetsIrisTestCsv, _checkpointPath, NetworkType.Onehot, 4, 3)
            {
                BatchSize = 10,
                EpochCheckpoint = 1,
                Epochs = 1000,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.00125f
            };

            ILayer[] layers =
            {
                new SimpleLayer(1, Activation.None),
            };

            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}