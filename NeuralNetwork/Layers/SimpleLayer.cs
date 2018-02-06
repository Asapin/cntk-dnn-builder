using System;
using CNTK;

namespace NeuralNetwork.Layers
{
    public class SimpleLayer : ILayer
    {
        private readonly int _outputDimesion;
        private readonly Activation.Apply _activation;

        public SimpleLayer(int outputDimesion, Activation.Apply activation)
        {
            _outputDimesion = outputDimesion;
            _activation = activation;
        }

        public Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            var inputVar = (Variable) input;
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                Math.Sqrt(1.0 / inputVar.Shape[0]),
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1);

            var shape = new[] { _outputDimesion, inputVar.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "weight");
            var biasParam = new Parameter(new NDShape(1 ,_outputDimesion), 0, device, "bias");

            var result = CNTKLib.Times(weightParam, inputVar) + biasParam;

            return _activation(result);
        }
    }
}