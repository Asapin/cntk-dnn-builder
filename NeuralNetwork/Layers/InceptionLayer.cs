using System.Collections.Generic;
using CNTK;
using NeuralNetwork.Layers.Descriptors;
using NeuralNetwork.Layers.Descriptors.Inception;

namespace NeuralNetwork.Layers
{
    public class InceptionLayer : AbstractLayer
    {
        private readonly Activation.Apply _activation;
        private readonly IEnumerable<ConvBlockDescriptor> _convBlockDescriptors;
        private readonly PoolingBlockDescriptor _poolingBlockDescriptor;
        private readonly StartingPoolingBlockDescriptor _startingPoolingBlockDescriptor;

        public InceptionLayer(Activation.Apply activation, IEnumerable<ConvBlockDescriptor> convBlockDescriptors, 
            PoolingBlockDescriptor poolingBlockDescriptor, StartingPoolingBlockDescriptor startingPoolingBlockDescriptor = null)
        {
            _activation = activation;
            _convBlockDescriptors = convBlockDescriptors;
            _poolingBlockDescriptor = poolingBlockDescriptor;
            _startingPoolingBlockDescriptor = startingPoolingBlockDescriptor;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device, string checkpointSavePath, bool log = true)
        {
            var start = input;

            if (_startingPoolingBlockDescriptor != null)
            {
                var layer = new PoolingLayer(_startingPoolingBlockDescriptor.Type,
                    _startingPoolingBlockDescriptor.KernelWidth, _startingPoolingBlockDescriptor.KernelHeight,
                    _startingPoolingBlockDescriptor.HStride, _startingPoolingBlockDescriptor.VStride);

                start = layer.Layer(ref input, ref device, checkpointSavePath, false);
            }
            
            var blocks = new List<Function>();
            foreach (var blockDescriptor in _convBlockDescriptors)
            {
                var layer1 = new ConvolutionLayer(Activation.None, 1, 1, blockDescriptor.ReaducedFiltersCount, 1, 1, true);
                var conv1X1 = layer1.Layer(ref start, ref device, checkpointSavePath, false);

                var layer2 = new ConvolutionLayer(Activation.None, blockDescriptor.KernelWidth, blockDescriptor.KernelHeight,
                    blockDescriptor.FiltersCount, blockDescriptor.HStride, blockDescriptor.VStride, true);
                var resultConv = layer2.Layer(ref conv1X1, ref device, checkpointSavePath, false);
                
                blocks.Add(resultConv);
            }

            var poolingLayer = new PoolingLayer(_poolingBlockDescriptor.Type, _poolingBlockDescriptor.KernelWidth,
                _poolingBlockDescriptor.KernelHeight, 1, 1, true);
            var pooling = poolingLayer.Layer(ref start, ref device, checkpointSavePath, false);

            var poolingConvLayer = new ConvolutionLayer(Activation.None, 1, 1, _poolingBlockDescriptor.FiltersCount, 1, 1, true);
            var poolingConv1X1 = poolingConvLayer.Layer(ref pooling, ref device, checkpointSavePath, false);
            blocks.Add(poolingConv1X1);

            var variableVector = new VariableVector();
            blocks.ForEach(block => variableVector.Add((Variable) block));

            var result = CNTKLib.Splice(variableVector, new Axis(2));

            LogShape(ref result, checkpointSavePath, "Inception", log);
            return _activation(result);
        }
    }
}