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
                new StreamConfiguration(descriptor.FeaturesStreamName, features.Shape[0]), 
                new StreamConfiguration(descriptor.LabelsStreamName, labels.Shape[0]) 
            };
            return MinibatchSource.TextFormatMinibatchSource(train ? descriptor.TrainDataPath : descriptor.TestDataPath, 
                streamConfig, MinibatchSource.InfinitelyRepeat, true);
        }
    }
}