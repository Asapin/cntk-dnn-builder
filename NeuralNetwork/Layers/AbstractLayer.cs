using System;
using CNTK;

namespace NeuralNetwork.Layers
{
    public abstract class AbstractLayer : ILayer
    {
        protected static CNTKDictionary GetGlorotUniformInitializer(ref Variable variable)
        {
            return CNTKLib.GlorotUniformInitializer(
                Math.Sqrt(1.0 / variable.Shape.TotalSize),
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1);
        }

        public abstract Function Layer(ref Function input, ref DeviceDescriptor device);
    }
}