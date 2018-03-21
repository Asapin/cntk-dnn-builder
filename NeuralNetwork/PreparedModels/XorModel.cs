using System.Collections.Generic;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for solving XOR problem.
    /// Current configuration is capable of achieving 100% accuracy on test data in ~10 epochs
    /// </summary>
    public class XorModel : AbstractModel
    {
        private const string XorTrainDataset = "../../Datasets/xor/train.csv";
        private const string XorTestDataset = "../../Datasets/xor/test.csv";

        public XorModel(string checkpointPath) : base(checkpointPath)
        {
        }

        protected override void Before()
        {
            //nothing
        }

        protected override NetworkDescriptor GetNetworkDescriptor()
        {
            return new NetworkDescriptor(XorTrainDataset, XorTestDataset, CheckpointPath, new[] { 2 }, 2)
            {
                BatchSize = 4,
                EvaluateFrequency = 1,
                Epochs = 100,
                CheckpointFrequency = 150,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRatePerSample = 0.125f
            };
        }

        protected override IEnumerable<ILayer> GetLayers()
        {
            ILayer[] layers =
            {
                new FullyConnectedLayer(Activation.ReLU, 8)
            };

            return layers;
        }
    }
}