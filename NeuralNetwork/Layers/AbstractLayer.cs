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

        protected static void LogShape(ref Function input, string layerName)
        {
            var variable = (Variable) input;
            var joinedShape = string.Join(" x ", variable.Shape.Dimensions);
            Console.WriteLine($"Layer: {layerName}, Shape: {joinedShape}, total neurons: {variable.Shape.TotalSize}");
        }

        public abstract Function Layer(ref Function input, ref DeviceDescriptor device);
    }
}