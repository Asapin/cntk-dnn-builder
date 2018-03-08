using CNTK;

namespace NeuralNetwork.Layers.Descriptors.Inception
{
    public class StartingPoolingBlockDescriptor
    {
        public StartingPoolingBlockDescriptor(PoolingType type, int kernelWidth, int kernelHeight, 
            int hStride = 1, int vStride = 1)
        {
            Type = type;
            KernelWidth = kernelWidth;
            KernelHeight = kernelHeight;
            HStride = hStride;
            VStride = vStride;
        }

        public PoolingType Type { get; }
        public int KernelWidth { get; }
        public int KernelHeight { get; }
        public int HStride { get; }
        public int VStride { get; }
    }
}