using System.Collections.Generic;
using NeuralNetwork.Layers;
using NeuralNetwork.Network;

namespace NeuralNetwork.PreparedModels
{
    public abstract class AbstractModel
    {
        protected readonly string CheckpointPath;

        protected AbstractModel(string checkpointPath)
        {
            CheckpointPath = checkpointPath;
        }

        protected abstract void Before();
        protected abstract NetworkDescriptor GetNetworkDescriptor();
        protected abstract IEnumerable<ILayer> GetLayers();

        public void Train()
        {
            Before();
            var descriptor = GetNetworkDescriptor();
            var layers = GetLayers();
            
            var network = new Network.NeuralNetwork(layers, descriptor);
            network.RunTraining();
        }
    }
}