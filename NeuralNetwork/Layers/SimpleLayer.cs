using CNTK;

namespace NeuralNetwork.Layers
{
    public class SimpleLayer : AbstractLayer
    {
        private readonly int _outputDimesion;
        private readonly Activation.Apply _activation;

        public SimpleLayer(int outputDimesion, Activation.Apply activation)
        {
            _outputDimesion = outputDimesion;
            _activation = activation;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var glorotInit = GetGlorotUniformInitializer(ref input);

            var shape = new[] { _outputDimesion, inputVar.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "weight");

            var result = CNTKLib.Times(weightParam, inputVar);
            var biasParam = new Parameter(new[] { NDShape.InferredDimension }, 0, device, "bias");
            result += biasParam;

            return _activation(result);
        }
    }
}