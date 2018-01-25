using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace NeuralNetwork.Network
{
    public class NetworkEvaluator
    {
        private readonly NetworkDescriptor _descriptor;

        public NetworkEvaluator(NetworkDescriptor descriptor)
        {
            _descriptor = descriptor;
        }
        
        public float EvaluateModel(ref Function networkModel, ref DeviceDescriptor device)
        {
            var features = networkModel.Arguments[0];
            var labels = networkModel.Output;

            var minibatchSource = NetworkUtils.GetMinibatchSource(_descriptor, false, ref features, ref labels);

            var featureStreamInfo = minibatchSource.StreamInfo(_descriptor.FeaturesStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(_descriptor.LabelsStreamName);

            var totalMisMatches = 0;
            var totalCount = 0L;

            while (true)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(_descriptor.BatchSize, device);

                totalCount += minibatchData[featureStreamInfo].numberOfSamples;

                var labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(labels);
                var expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                var inputDataMap = new Dictionary<Variable, Value>
                {
                    {features, minibatchData[featureStreamInfo].data}
                };

                var outputDataMap = new Dictionary<Variable, Value>
                {
                    {labels, null}
                };
                
                networkModel.Evaluate(inputDataMap, outputDataMap, device);
                var outputData = outputDataMap[labels].GetDenseData<float>(labels);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

#if DEBUG
                foreach (var ints in outputData)
                {
                    foreach (var i1 in ints)
                    {
                        Console.Write($"{i1} ");
                    }
                    Console.WriteLine();
                }
                
                Console.Write("Expected: ");
                expectedLabels.ForEach(Console.Write);
                Console.WriteLine();
                
                Console.Write("Actual: ");
                actualLabels.ForEach(Console.Write);
                Console.WriteLine();
                Console.WriteLine("================");
#endif

                var misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();
                totalMisMatches += misMatches;
                
                if (!minibatchData.Values.Any(a => a.sweepEnd)) continue;
                break;
            }

            return 1.0F - (float) totalMisMatches / totalCount;
        }
    }
}