using CNTK;

namespace NeuralNetwork.Layers.Descriptors.Inception
{
    public class PoolingBlockDescriptor
    {
        public PoolingBlockDescriptor(PoolingType type, int kernelWidth, int kernelHeight, int filtersCount)
        {
            Type = type;
            KernelWidth = kernelWidth;
            KernelHeight = kernelHeight;
            FiltersCount = filtersCount;
        }

        public PoolingType Type { get; }
        public int KernelWidth { get; }
        public int KernelHeight { get; }
        public int FiltersCount { get; }
    }
}