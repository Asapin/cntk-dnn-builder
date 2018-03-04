using CNTK;

namespace NeuralNetwork.Network
{
    public abstract class NetworkUtils
    {
        public static  MinibatchSource GetMinibatchSource(NetworkDescriptor descriptor, bool train, 
            ref Variable features, ref Variable labels)
        {
            var streamConfig = new[]
            {
                new StreamConfiguration(descriptor.FeaturesStreamName, features.Shape.TotalSize), 
                new StreamConfiguration(descriptor.LabelsStreamName, labels.Shape.TotalSize) 
            };
            return MinibatchSource.TextFormatMinibatchSource(train ? descriptor.TrainDataPath : descriptor.TestDataPath, 
                streamConfig, MinibatchSource.InfinitelyRepeat, true);
        }
    }
}