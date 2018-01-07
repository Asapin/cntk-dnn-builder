using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;

namespace NeuralNetwork
{
    internal static class Program
    {
        private static readonly int[] Layers = {12, 6};
        private const int Epochs = 100;
        private const string FeaturesStreamName = "features";
        private const string LabelsStreamName = "labels";
        private const int InputDimension = 189;
        private const int OutputClasses = 21;
        private const uint BatchSize = 100;

        public static void Main(string[] args)
        {
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

            var ffnnModel = CreateFastForwardNetwork(ref feature, Activation.Sigmoid, OutputClasses, "stocks", ref device, Layers);

            var trainigLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(ffnnModel), labels, "LossFunction");
            var classificationError = CNTKLib.ClassificationError(new Variable(ffnnModel), labels, "classificationError");

            var sgdLearner = Learner.SGDLearner(ffnnModel.Parameters(), learningRatePerSample);
            var trainer = Trainer.CreateTrainer(ffnnModel, trainigLoss, classificationError, new List<Learner> {sgdLearner});

            var minEvaluation = double.MaxValue;
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
                var evaluationAverage = trainer.PreviousMinibatchEvaluationAverage();
                if (evaluationAverage < minEvaluation)
                {
                    minEvaluation = evaluationAverage;
                    DumpNetwork(evaluationAverage, i);
                } else if (i % 100 == 0)
                {
                    DumpNetwork(evaluationAverage, i);
                }
            }
        }

        private static Function CreateFastForwardNetwork(ref Variable input, Activation activation, int outputDimension, 
            string modelName, ref DeviceDescriptor device, IEnumerable<int> layers)
        {
            var h = (Function)input;
            
            foreach (var layer in layers)
            {
                if (layer == 0) continue;
                h = SimpleLayer(h, layer * 10, ref device);
                h = ApplyActivationFunction(ref h, activation);
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

        private static void DumpNetwork(double evaluationAverage, int epoch)
        {
            var acc = Math.Round((1.0 - evaluationAverage) * 100, 2);
            Console.WriteLine($"Time: {DateTime.Now}; Epoch: {epoch}; Accuracy: {acc}%");
        }
    }
}