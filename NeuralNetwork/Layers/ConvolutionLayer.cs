using CNTK;

namespace NeuralNetwork.Layers
{
    public class ConvolutionLayer : AbstractLayer
    {        
        private readonly Activation.Apply _activation;
        private readonly int _kernelWidth;
        private readonly int _kernelHeight;
        private readonly int _filtersCount;
        private readonly int _hStride;
        private readonly int _vStride;
        private readonly bool _padding;

        public ConvolutionLayer(Activation.Apply activation, int kernelWidth, int kernelHeight, 
            int filtersCount, int hStride = 1, int vStride = 1, bool padding = false)
        {
            _activation = activation;
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _filtersCount = filtersCount;
            _hStride = hStride;
            _vStride = vStride;
            _padding = padding;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var glorotInit = GetGlorotUniformInitializer(ref inputVar);

            var numInputChannels = inputVar.Shape[inputVar.Shape.Rank - 1];

            var convParams = new Parameter(new[] { _kernelWidth, _kernelHeight, numInputChannels, _filtersCount }, 
                DataType.Float, glorotInit, device, "convParams");

            var result = CNTKLib.Convolution(convParams, inputVar, new[] { _hStride, _vStride, numInputChannels }, 
                new[] { true }, new[] { _padding });

            LogShape(ref result, "Convolution");

            return _activation(result);
        }
    }
}