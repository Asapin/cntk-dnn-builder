using System.Collections.Generic;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    /// <summary>
    /// This is an example model for learning Iris dataset.
    /// Current configuration is capable of achieving 100% accuracy on test data.
    /// </summary>
    public class IrisModel : AbstractModel
    {
        private const string IrisTrainDataset = "../../Datasets/iris/train.csv";
        private const string IrisTestDataset = "../../Datasets/iris/test.csv";

        public IrisModel(string checkpointPath) : base(checkpointPath)
        {
        }

        protected override void Before()
        {
            //nothing
        }

        protected override NetworkDescriptor GetNetworkDescriptor()
        {
            return new NetworkDescriptor(IrisTrainDataset, IrisTestDataset, CheckpointPath, 
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
            };;
        }

        protected override IEnumerable<ILayer> GetLayers()
        {
            ILayer[] layers =
            {
                new FullyConnectedLayer(Activation.LeakyReLU, 9)
            };

            return layers;
        }
    }
}