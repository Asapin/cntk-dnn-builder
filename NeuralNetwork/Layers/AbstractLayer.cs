using System;
using CNTK;

namespace NeuralNetwork.Layers
{
    public abstract class AbstractLayer : ILayer
    {
        public CNTKDictionary GetGlorotUniformInitializer(ref Function input)
        {
            var inputVar = (Variable) input;
            return CNTKLib.GlorotUniformInitializer(
                Math.Sqrt(1.0 / inputVar.Shape.TotalSize),
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1);
        }

        public abstract Function Layer(ref Function input, ref DeviceDescriptor device);
    }
}