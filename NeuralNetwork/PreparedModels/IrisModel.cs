﻿using System.Collections.Generic;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;
using NeuralNetwork.Network.LearningRates;

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
            return new NetworkDescriptor(IrisTrainDataset, IrisTestDataset, CheckpointPath, new[] { 4 }, 3)
            {
                BatchSize = 10,
                EvaluationFrequency = 1,
                CheckpointFrequency = 150,
                EpochsToTrain = 100,
                Evaluate = true,
                FeaturesStreamName = "features",
                LabelsStreamName = "labels",
                LearningRate = new StaticLearningRate(0.0015f)
            };
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