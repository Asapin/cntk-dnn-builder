using CNTK;

namespace NeuralNetwork.Layers
{
    public class BatchNormalizationLayer : AbstractLayer
    {
        private readonly Activation.Apply _activation;
        private readonly bool _spatial;
        private readonly int _bnTimeConst;

        public BatchNormalizationLayer(Activation.Apply activation, bool spatial, int bnTimeConst)
        {
            _activation = activation;
            _spatial = spatial;
            _bnTimeConst = bnTimeConst;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var glorotInit = GetGlorotUniformInitializer(ref input);

            var biasParams = new Parameter(new[] { NDShape.InferredDimension }, DataType.Float, glorotInit, device);
            var scaleParams = new Parameter(new[] { NDShape.InferredDimension }, DataType.Float, glorotInit, device);
            var runningMean = new Constant(new[] { NDShape.InferredDimension }, 0.0f, device);
            var runningInvStd = new Constant(new[] { NDShape.InferredDimension }, 0.0f, device);
            var runningCount = Constant.Scalar(0.0f, device);

            var batchNormalizationLayer = CNTKLib.BatchNormalization(input, scaleParams, biasParams, 
                runningMean, runningInvStd, runningCount, _spatial, _bnTimeConst, 0.0, 1e-5 /* epsilon */);

            return _activation(batchNormalizationLayer);
        }
    }
}