using CNTK;

namespace NeuralNetwork.Layers
{
    public interface ILayer
    {
        Function Layer(ref Function input, ref DeviceDescriptor device, string checkpointSavePath, bool log = true);
    }
}