namespace NeuralNetwork.Layers.Descriptors.Inception
{
    public class ConvBlockDescriptor
    {
        public ConvBlockDescriptor(int readucedFiltersCount, int kernelWidth, int kernelHeight, int filtersCount, 
            int hStride = 1, int vStride = 1)
        {
            ReaducedFiltersCount = readucedFiltersCount;
            KernelWidth = kernelWidth;
            KernelHeight = kernelHeight;
            FiltersCount = filtersCount;
            HStride = hStride;
            VStride = vStride;
        }

        public int ReaducedFiltersCount { get; }
        public int KernelWidth { get; }
        public int KernelHeight { get; }
        public int FiltersCount { get; }
        public int HStride { get; }
        public int VStride { get; }
    }
}