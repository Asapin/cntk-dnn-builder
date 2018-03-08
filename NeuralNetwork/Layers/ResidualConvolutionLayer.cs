using CNTK;

namespace NeuralNetwork.Layers
{
    public class ResidualConvolutionLayer : AbstractLayer
    {
        private readonly Activation.Apply _activation;
        private readonly int _kernelWidth;
        private readonly int _kernelHeight;
        private readonly int _filtersCount;
        private readonly int _hStride;
        private readonly int _vStride;

        public ResidualConvolutionLayer(Activation.Apply activation, int kernelWidth, int kernelHeight, 
            int filtersCount, int hStride = 1, int vStride = 1)
        {
            _activation = activation;
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _filtersCount = filtersCount;
            _hStride = hStride;
            _vStride = vStride;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device, string checkpointSavePath, bool log = true)
        {
            var layer1 = GetLayer(ref input, ref device);
            var result1 = _activation(layer1);

            Function result;
            if (result1.Output.Shape.Equals(input.Output.Shape))
            {
                var layer2 = GetLayer(ref result1, ref device);
                result = layer2 + (Variable) input;
            }
            else
            {
                var layer2 = GetLayer(ref result1, ref device);
                var result2 = _activation(layer2);

                var layer3 = GetLayer(ref result2, ref device);
                result = layer3 + (Variable) result1;
            }

            LogShape(ref result, checkpointSavePath, "Residual Convolution", log);
            return _activation(result);
        }

        private Function GetLayer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var glorotInit = GetGlorotUniformInitializer(ref inputVar);

            var numInputChannels = inputVar.Shape[inputVar.Shape.Rank - 1];

            var convParams = new Parameter(new[] { _kernelWidth, _kernelHeight, numInputChannels, _filtersCount }, 
                DataType.Float, glorotInit, device, "convParams");

            return CNTKLib.Convolution(convParams, inputVar, new[] { _hStride, _vStride, numInputChannels }, 
                new[] { true }, new[] { true });
        }
    }
}