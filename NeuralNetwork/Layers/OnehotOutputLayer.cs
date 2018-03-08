﻿using CNTK;

namespace NeuralNetwork.Layers
{
    public class OnehotOutputLayer : AbstractLayer
    {
        private readonly int _outputDimesion;

        public OnehotOutputLayer(int outputDimesion)
        {
            _outputDimesion = outputDimesion;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device, string checkpointSavePath, bool log = true)
        {
            var inputVar = (Variable) input;
            var glorotInit = GetGlorotUniformInitializer(ref inputVar);

            var shape = new[] { _outputDimesion, inputVar.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "weight");
            var biasParam = new Parameter(new NDShape(1 ,_outputDimesion), 0, device, "bias");

            var result = CNTKLib.Times(weightParam, inputVar) + biasParam;
            LogShape(ref result, checkpointSavePath, "Onehot", log);

            return result;
        }
    }
}