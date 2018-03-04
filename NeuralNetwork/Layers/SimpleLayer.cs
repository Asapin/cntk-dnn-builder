using CNTK;

namespace NeuralNetwork.Layers
{
    public class SimpleLayer : AbstractLayer
    {
        private readonly int _outputDimesion;
        private readonly Activation.Apply _activation;
        private readonly bool _applyBatchNorm;
        private readonly bool _spatial;

        public SimpleLayer(int outputDimesion, Activation.Apply activation, bool applyBatchNorm = false, bool spatial = false)
        {
            _outputDimesion = outputDimesion;
            _activation = activation;
            _applyBatchNorm = applyBatchNorm;
            _spatial = spatial;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var glorotInit = GetGlorotUniformInitializer(ref input);

            var shape = new[] { _outputDimesion, inputVar.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "weight");

            var result = CNTKLib.Times(weightParam, inputVar);
            if (_applyBatchNorm)
            {
                var betaParam = new Parameter(new[] { NDShape.InferredDimension }, 0, device, "beta");
                var gammaParam = new Parameter(new[] { NDShape.InferredDimension }, 0, device, "gamma");
                var runningMean = new Constant(new[] { NDShape.InferredDimension }, 0.0f, device, "runningMean");
                var runningInvStd = new Constant(new[] { NDShape.InferredDimension }, 0.0f, device, "runningInvStd");
                var runningCount = Constant.Scalar(0.0f, device);

                result = CNTKLib.BatchNormalization(result, gammaParam, betaParam, runningMean, runningInvStd,
                    runningCount, _spatial);
            }
            else
            {
                var biasParam = new Parameter(new[] { NDShape.InferredDimension }, 0, device, "bias");
                result += biasParam;
            }

            return _activation(result);
        }
    }
}