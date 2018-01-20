using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using NeuralNetwork.Layers;

namespace NeuralNetwork
{
    internal static class Program
    {
        private const string DateTimeFormat = "yyyy-MM-dd-HH-mm";

        private const int Epochs = 5000;
        private const string FeaturesStreamName = "features";
        private const string LabelsStreamName = "labels";
        private const int InputDimension = 189;
        private const int OutputClasses = 3;
        private const uint BatchSize = 10;

        public static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("You should provide path to a folder with train and test data");
                return;
            }
            var defaultDevice = DeviceDescriptor.UseDefaultDevice();
            Console.WriteLine($"CNTK, using {defaultDevice.Type}");
            
            var trainPath = Path.Combine(args[0], "train.csv");
            var testPath = Path.Combine(args[0], "test.csv");
            TrainNn(ref defaultDevice, trainPath, testPath, args[1]);
        }

        private static void TrainNn(ref DeviceDescriptor device, string trainPath, string testPath, string saveDir)
        {
            var features = Variable.InputVariable(new[] { InputDimension }, DataType.Float, FeaturesStreamName);
            var labels = Variable.InputVariable(new[] { OutputClasses }, DataType.Float, LabelsStreamName);

            ILayer[] layers =
            {
                new SimpleLayer(200, Activation.ReLU),
                new SimpleLayer(100, Activation.ReLU),
                new SimpleLayer(OutputClasses, Activation.Sigmoid),
            };

            Function classifierOutput = features;
            foreach (var layer in layers)
            {
                classifierOutput = layer.Layer(ref classifierOutput, ref device);
            }

            var minibatchSource = GetMinibatchSource(trainPath, ref features, ref labels);
            var trainer = GetTrainer(ref classifierOutput, ref labels);

            var featureStreamInfo = minibatchSource.StreamInfo(FeaturesStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(LabelsStreamName);

            var i = 0;
            var savePath = GetSavePath(saveDir);
            while (i <= Epochs)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(BatchSize, device);

                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    {features, minibatchData[featureStreamInfo]},
                    {labels, minibatchData[labelStreamInfo]}
                };

                trainer.TrainMinibatch(arguments, device);

                if (!minibatchData.Values.Any(a => a.sweepEnd)) continue;

                i++;
                DumpNetwork(ref classifierOutput, ref device, ref trainer, testPath, i, savePath);
            }
        }

        private static void DumpNetwork(ref Function networkModel, ref DeviceDescriptor device, ref Trainer trainer,
            string testPath, int epoch, string savePath)
        {
            var accuracy = EvaluateModel(ref networkModel, ref device, testPath);

            var info = $"{DateTime.Now}; {epoch}; {accuracy}; {trainer.PreviousMinibatchLossAverage()}";
            Console.WriteLine(info);
            File.AppendAllLines(Path.Combine(savePath, "info.csv"), new []{ info });

            if (epoch % 10 != 0) return;

            var epochPath = Path.Combine(savePath, epoch.ToString());
            trainer.SaveCheckpoint(Path.Combine(epochPath, "model"));
        }
        
        private static float EvaluateModel(ref Function networkModel, ref DeviceDescriptor device, string testPath)
        {
            var features = networkModel.Arguments[0];
            var labels = networkModel.Output;

            var minibatchSource = GetMinibatchSource(testPath, ref features, ref labels);

            var featureStreamInfo = minibatchSource.StreamInfo(FeaturesStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(LabelsStreamName);

            var totalMisMatches = 0;
            var totalCount = 0L;

            while (true)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(BatchSize, device);

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

                var misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();
                totalMisMatches += misMatches;
                
                if (!minibatchData.Values.Any(a => a.sweepEnd)) continue;
                break;
            }

            return 1.0F - (float) totalMisMatches / totalCount;
        }

        private static string GetSavePath(string saveDir)
        {
            var dateTimeString = DateTime.Now.ToString(DateTimeFormat);
            var savePath = Path.Combine(saveDir, dateTimeString);
            Directory.CreateDirectory(savePath);

            return savePath;
        }

        private static Trainer GetTrainer(ref Function model, ref Variable labels)
        {
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(model), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(model), labels, "classificationError");

            var learners = GetLearners(ref model);

            return Trainer.CreateTrainer(model, trainingLoss, prediction, learners);
        }

        private static IList<Learner> GetLearners(ref Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.001125, 1);
            return new List<Learner>() {Learner.SGDLearner(model.Parameters(), learningRatePerSample)};
        }

        private static MinibatchSource GetMinibatchSource(string sourcePath, ref Variable features, ref Variable labels)
        {
            var streamConfig = new[]
            {
                new StreamConfiguration(FeaturesStreamName, features.Shape[0]), 
                new StreamConfiguration(LabelsStreamName, labels.Shape[0]) 
            };
            return MinibatchSource.TextFormatMinibatchSource(sourcePath, streamConfig,
                MinibatchSource.InfinitelyRepeat, true);
        }
    }
}