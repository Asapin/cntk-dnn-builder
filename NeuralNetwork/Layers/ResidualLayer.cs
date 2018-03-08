using CNTK;

namespace NeuralNetwork.Layers
{
    public class ResidualLayer : AbstractLayer
    {
        private readonly Activation.Apply _activation;

        public ResidualLayer(Activation.Apply activation)
        {
            _activation = activation;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device, string checkpointSavePath)
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

            LogShape(ref result, checkpointSavePath, "Residual");
            return _activation(result);
        }

        private static Function GetLayer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var glorotInit = GetGlorotUniformInitializer(ref inputVar);

            var shape = new[] { inputVar.Shape[0], inputVar.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "weight");

            var result = CNTKLib.Times(weightParam, inputVar);
            var biasParam = new Parameter(new[] { NDShape.InferredDimension }, 0, device, "bias");
            result += biasParam;

            return result;
        }
    }
}