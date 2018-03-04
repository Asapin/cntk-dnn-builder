using System.Linq;
using CNTK;

namespace NeuralNetwork.Layers
{
    public class FullyConnectedLayer : AbstractLayer
    {
        private readonly int _outputDimesion;
        private readonly Activation.Apply _activation;

        public FullyConnectedLayer(Activation.Apply activation, int outputDimesion)
        {
            _activation = activation;
            _outputDimesion = outputDimesion;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            if (inputVar.Shape.Rank != 1)
            {
                var newDim = inputVar.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                inputVar = CNTKLib.Reshape(inputVar, new[] { newDim });
            }

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