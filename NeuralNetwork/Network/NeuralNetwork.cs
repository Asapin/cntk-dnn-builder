﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using NeuralNetwork.Layers;

namespace NeuralNetwork.Network
{
    public class NeuralNetwork
    {
        private readonly IEnumerable<ILayer> _layers;
        private readonly NetworkDescriptor _descriptor;
        private readonly NetworkEvaluator _evaluator;

        public NeuralNetwork(IEnumerable<ILayer> layers, NetworkDescriptor descriptor)
        {
            _layers = layers;
            _descriptor = descriptor;
            _evaluator = new NetworkEvaluator(descriptor);
        }

        public void RunTraining()
        {
            var device = DeviceDescriptor.UseDefaultDevice();
            Console.WriteLine($"CNTK, using {device.Type}");
            
            var features = Variable.InputVariable(new[] { _descriptor.InputDimension }, DataType.Float, _descriptor.FeaturesStreamName);
            var labels = Variable.InputVariable(new[] { _descriptor.OutputClasses }, DataType.Float, _descriptor.LabelsStreamName);

            var classifierOutput = GetModel(ref features, ref device);

            var minibatchSource = NetworkUtils.GetMinibatchSource(_descriptor, true, ref features, ref labels);
            var trainer = GetTrainer(ref classifierOutput, ref labels);

            var featureStreamInfo = minibatchSource.StreamInfo(_descriptor.FeaturesStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(_descriptor.LabelsStreamName);

            var i = 0;
            while (i <= _descriptor.Epochs)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(_descriptor.BatchSize, device);

                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    {features, minibatchData[featureStreamInfo]},
                    {labels, minibatchData[labelStreamInfo]}
                };

                trainer.TrainMinibatch(arguments, device);

                if (!minibatchData.Values.Any(a => a.sweepEnd)) continue;

                i++;
                if (i % _descriptor.EpochCheckpoint == 0)
                {
                    DumpNetwork(ref classifierOutput, ref device, ref trainer, i);
                }
            }
        }
        
        private void DumpNetwork(ref Function networkModel, ref DeviceDescriptor device, ref Trainer trainer, int epoch)
        {
            var accuracy = float.NaN;
            if (_descriptor.Evaluate)
            {
                accuracy = _evaluator.EvaluateModel(ref networkModel, ref device);
            }

            var info = $"{DateTime.Now}; {epoch}; {accuracy}; {trainer.PreviousMinibatchLossAverage()}";
            Console.WriteLine(info);

            File.AppendAllLines(Path.Combine(_descriptor.CheckpointSavePath, "info.csv"), new []{ info });

            if (epoch % 10 != 0) return;

            var epochPath = Path.Combine(_descriptor.CheckpointSavePath, epoch.ToString());
            trainer.SaveCheckpoint(Path.Combine(epochPath, "model"));
        }

        private Trainer GetTrainer(ref Function model, ref Variable labels)
        {
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(model), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(model), labels, "classificationError");

            var learners = GetLearners(ref model);

            return Trainer.CreateTrainer(model, trainingLoss, prediction, learners);
        }

        private IList<Learner> GetLearners(ref Function model)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(_descriptor.LearningRatePerSample, 1);
            return new List<Learner> {Learner.SGDLearner(model.Parameters(), learningRatePerSample)};
        }

        private Function GetModel(ref Variable features, ref DeviceDescriptor device)
        {
            Function model = features;
            foreach (var layer in _layers)
            {
                model = layer.Layer(ref model, ref device);
            }

            switch (_descriptor.Type)
            {
                case NetworkType.Onehot: 
                    var onehotLayer = new OnehotOutputLayer(_descriptor.OutputClasses);
                    return onehotLayer.Layer(ref model, ref device);
                case NetworkType.Regression:
                    var regressionLayer = new RegressionOutputLayer(_descriptor.OutputClasses);
                    return regressionLayer.Layer(ref model, ref device);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}