using CNTK;

namespace NeuralNetwork.Layers
{
    public class PoolingLayer : AbstractLayer
    {
        private readonly PoolingType _poolingType;
        private readonly int _kernelWidth;
        private readonly int _kernelHeight;
        private readonly int _hStride;
        private readonly int _vStride;
        private readonly bool _padding;

        public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight, 
            int hStride = 1, int vStride = 1, bool padding = false)
        {
            _poolingType = poolingType;
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _hStride = hStride;
            _vStride = vStride;
            _padding = padding;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device, string checkpointSavePath, bool log = true)
        {
            var inputVar = (Variable) input;
            var result = CNTKLib.Pooling(inputVar, _poolingType, new[] { _kernelWidth, _kernelHeight }, 
                new[] { _hStride, _vStride }, new[] { _padding });
            
            LogShape(ref result, checkpointSavePath, "Pooling", log);
            return result;
        }
    }
}