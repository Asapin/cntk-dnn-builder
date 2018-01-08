using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;

namespace NeuralNetwork
{
    internal static class Program
    {
        private static readonly LayerConfiguration[] Layers =
        {
            new LayerConfiguration(100, Activation.ReLU),
            new LayerConfiguration(50, Activation.ReLU)
        };
        private const int Epochs = 5000;
        private const string FeaturesStreamName = "features";
        private const string LabelsStreamName = "labels";
        private const int InputDimension = 189;
        private const int OutputClasses = 3;
        private const uint BatchSize = 100;

        public static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("You should provide path to a folder with train and test data");
                return;
            }
            var defaultDevice = DeviceDescriptor.UseDefaultDevice();
            Console.WriteLine($"CNTK, using {defaultDevice.Type}");
            
            var trainPath = Path.Combine(args[0], "train.csv");
            var testPath = Path.Combine(args[0], "test.csv");
            TrainNn(ref defaultDevice, trainPath, testPath);
        }

        private static void TrainNn(ref DeviceDescriptor device, string trainPath, string testPath)
        {
            var streamConfig = new[]
            {
                new StreamConfiguration(FeaturesStreamName, InputDimension),
                new StreamConfiguration(LabelsStreamName, OutputClasses)
            };

            var feature = Variable.InputVariable(new NDShape(1, InputDimension), DataType.Float, FeaturesStreamName);
            var labels = Variable.InputVariable(new NDShape(1, OutputClasses), DataType.Float, LabelsStreamName);

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(trainPath, streamConfig,
                MinibatchSource.InfinitelyRepeat, true);

            var featureStreamInfo = minibatchSource.StreamInfo(FeaturesStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(LabelsStreamName);
            
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.001125, 1);

            var ffnnModel = CreateFastForwardNetwork(ref feature, OutputClasses, "stocks", ref device, Layers);

            var trainigLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(ffnnModel), labels, "LossFunction");
            var classificationError = CNTKLib.ClassificationError(new Variable(ffnnModel), labels, "classificationError");

            var sgdLearner = Learner.SGDLearner(ffnnModel.Parameters(), learningRatePerSample);
            var trainer = Trainer.CreateTrainer(ffnnModel, trainigLoss, classificationError, new List<Learner> {sgdLearner});

            var minLossAverage = double.MaxValue;
            var i = 0;
            while (i <= Epochs)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(BatchSize, device);

                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    {feature, minibatchData[featureStreamInfo]},
                    {labels, minibatchData[labelStreamInfo]}
                };

                trainer.TrainMinibatch(arguments, device);

                if (!minibatchData.Values.Any(a => a.sweepEnd)) continue;

                i++;
                var lossAverage = trainer.PreviousMinibatchLossAverage();
                if (lossAverage < minLossAverage)
                {
                    minLossAverage = lossAverage;
                    Console.WriteLine($"Time: {DateTime.Now}; Epoch: {i}; Found new minimal loss: {minLossAverage}");
                } else if (i % 100 == 0)
                {
                    DumpNetwork(ref ffnnModel, ref device, testPath, i);
                }
            }
            
            Console.WriteLine($"Time: {DateTime.Now}; Minimal average loss: {minLossAverage}");
        }

        private static Function CreateFastForwardNetwork(ref Variable input, int outputDimension, 
            string modelName, ref DeviceDescriptor device, IEnumerable<LayerConfiguration> layers)
        {
            var h = (Function)input;
            
            foreach (var layer in layers)
            {
                h = SimpleLayer(h, layer.NeuronCount, ref device);
                h = ApplyActivationFunction(ref h, layer.Activation);
            }

            var output = SimpleLayer(h, outputDimension, ref device);
            output.SetName(modelName);
            return output;
        }

        private static Function SimpleLayer(Function input, int outputDimension, ref DeviceDescriptor device)
        {
            var glorotInitializer = CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                1);

            var variable = (Variable) input;
            var shape = new[] { outputDimension, variable.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInitializer, device, "w");
            var biasParam = new Parameter(new NDShape(1, outputDimension), 0, device, "b");

            return CNTKLib.Times(weightParam, input) + biasParam;
        }

        private static Function ApplyActivationFunction(ref Function layer, Activation activation)
        {
            switch (activation)
            {
                default: return layer;
                case Activation.ReLU: return CNTKLib.ReLU(layer);
                case Activation.Sigmoid: return CNTKLib.Sigmoid(layer);
                case Activation.Tanh: return CNTKLib.Tanh(layer);
            }
        }

        private static void DumpNetwork(ref Function networkModel, ref DeviceDescriptor device, string testPath, int epoch)
        {
            Console.WriteLine($"Time: {DateTime.Now}; Epoch: {epoch}; Evaluating model...");
            var accuracy = EvaluateModel(ref networkModel, ref device, testPath);
            Console.WriteLine($"Accuracy: {accuracy * 100}%");
            
        }
        
        private static float EvaluateModel(ref Function networkModel, ref DeviceDescriptor device, string testPath)
        {
            var feature = networkModel.Arguments[0];
            var label = networkModel.Output;

            var streamConfiguration = new[]
            {
                new StreamConfiguration(FeaturesStreamName, feature.Shape[0]), 
                new StreamConfiguration(LabelsStreamName, label.Shape[0]) 
            };

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(testPath, streamConfiguration, 
                MinibatchSource.InfinitelyRepeat, true);

            var featureStreamInfo = minibatchSource.StreamInfo(FeaturesStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(LabelsStreamName);

            var totalMisMatches = 0;
            var totalCount = 0L;

            while (true)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(BatchSize, device);
                if (minibatchData == null || minibatchData.Count == 0)
                {
                    break;
                }

                totalCount += minibatchData[featureStreamInfo].numberOfSamples;

                var labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(label);
                var expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                var inputDataMap = new Dictionary<Variable, Value>()
                {
                    {feature, minibatchData[featureStreamInfo].data}
                };

                var outputDataMap = new Dictionary<Variable, Value>()
                {
                    {label, null}
                };
                
                networkModel.Evaluate(inputDataMap, outputDataMap, device);
                var outputData = outputDataMap[label].GetDenseData<float>(label);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                var misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();
                totalMisMatches += misMatches;
                
                if (!minibatchData.Values.Any(a => a.sweepEnd)) continue;
                break;
            }

            return 1.0F - (float) totalMisMatches / totalCount;
        }
    }
}