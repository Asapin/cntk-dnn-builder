using System;
using System.IO;
using NeuralNetwork.Network.LearningRates;

namespace NeuralNetwork.Network
{
    public class NetworkDescriptor
    {
        private const string DateTimeFormat = "yyyy-MM-dd-HH-mm";
        private readonly DateTime _dateTime = DateTime.Now;

        public NetworkDescriptor(string trainDataPath, string testDataPath, string checkpointPath,
            int[] inputDimension, int outputClasses)
        {
            TrainDataPath = trainDataPath;
            TestDataPath = testDataPath;
            CheckpointPath = checkpointPath;
            InputDimension = inputDimension;
            OutputClasses = outputClasses;
        }

        public int? EpochsToTrain { get; set; }
        public uint BatchSize { get; set; } = 100;
        public bool Evaluate { get; set; } = true;
        public int EvaluationFrequency { get; set; } = 10;
        public int CheckpointFrequency { get; set; } = 10;
        public string FeaturesStreamName { get; set; } = "features";
        public string LabelsStreamName { get; set; } = "labels";
        public ILearningRate LearningRate { get; set; } = new StaticLearningRate(0.00125f);
        public float MomentumPerSample { get; set; } = 0.9f;
        public float L2RegularizationWeight { get; set; } = float.NaN;
        public string TrainDataPath { get; }
        public string TestDataPath { get; }
        public string ModelCheckpointPath { get; set;  }
        private string CheckpointPath { get; }
        public NetworkType Type { get; set; } = NetworkType.Onehot;
        public int[] InputDimension { get; }
        public int OutputClasses { get; }

        public string CheckpointSavePath
        {
            get
            {
                var dateTimeString = _dateTime.ToString(DateTimeFormat);
                var savePath = Path.Combine(CheckpointPath, dateTimeString);
                if (!Directory.Exists(savePath))
                {
                    Directory.CreateDirectory(savePath);                    
                }

                return savePath;
            }
        }
    }
}