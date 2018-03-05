using CNTK;

namespace NeuralNetwork.Layers
{
    public class PoolingLayer : AbstractLayer
    {
        private readonly PoolingType _poolingType;
        private readonly int _hFilterSize;
        private readonly int _vFilterSize;
        private readonly int _hStride;
        private readonly int _vStride;
        private readonly bool _padding;

        public PoolingLayer(PoolingType poolingType, int hFilterSize, int vFilterSize, 
            int hStride = 1, int vStride = 1, bool padding = false)
        {
            _poolingType = poolingType;
            _hFilterSize = hFilterSize;
            _vFilterSize = vFilterSize;
            _hStride = hStride;
            _vStride = vStride;
            _padding = padding;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var result = CNTKLib.Pooling(inputVar, _poolingType, new[] { _hFilterSize, _vFilterSize }, 
                new[] { _hStride, _vStride }, new[] { _padding });
            
            LogShape(ref result, "Pooling");
            return result;
        }
    }
}