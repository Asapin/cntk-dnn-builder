using CNTK;

namespace NeuralNetwork.Layers
{
    public class OnehotOutputLayer : ILayer
    {
        private readonly int _outputDimesion;

        public OnehotOutputLayer(int outputDimesion)
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

            var result = CNTKLib.Times(weightParam, inputVar) + biasParam;

            return result;
        }
    }
}