using CNTK;

namespace NeuralNetwork.Layers
{
    public class RegressionOutputLayer : ILayer
    {
        private readonly int _outputDimesion;

        public RegressionOutputLayer(int outputDimesion)
        {
            _outputDimesion = outputDimesion;
        }

        public Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                0.01,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1);
            
            var inputVar = (Variable) input;

            var shape = new[] { _outputDimesion, inputVar.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "weight");
            var biasParam = new Parameter(new NDShape(1 ,_outputDimesion), 0, device, "bias");

            var multiply = CNTKLib.Times(weightParam, inputVar);
            var result = CNTKLib.Plus(multiply, biasParam);

            return result;
        }
    }
}