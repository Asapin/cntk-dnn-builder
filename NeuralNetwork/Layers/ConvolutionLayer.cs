using CNTK;

namespace NeuralNetwork.Layers
{
    public class ConvolutionLayer : ILayer
    {        
        private readonly Activation.Apply _activation;
        private readonly int _kernelWidth;
        private readonly int _kernelHeight;
        private readonly int _outFeatureMapCount;
        private readonly int _hStride;
        private readonly int _vStride;

        public ConvolutionLayer(Activation.Apply activation, int kernelWidth, int kernelHeight, 
            int outFeatureMapCount, int hStride, int vStride)
        {
            _activation = activation;
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _outFeatureMapCount = outFeatureMapCount;
            _hStride = hStride;
            _vStride = vStride;
        }

        public Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;

            var numInputChannels = inputVar.Shape[inputVar.Shape.Rank - 1];
            
            var initializer = CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1);

            var convParams = new Parameter(new[] { _kernelWidth, _kernelHeight, numInputChannels, _outFeatureMapCount },
                DataType.Float, initializer, device);

            var convFunction = CNTKLib.Convolution(convParams, inputVar, new[] { _hStride, _vStride, numInputChannels });

            return _activation(convFunction);
        }
    }
}