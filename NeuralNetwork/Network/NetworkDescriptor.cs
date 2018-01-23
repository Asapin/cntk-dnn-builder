﻿using System;
using System.IO;

namespace NeuralNetwork.Network
{
    public class NetworkDescriptor
    {
        private const string DateTimeFormat = "yyyy-MM-dd-HH-mm";
        private readonly DateTime _dateTime = DateTime.Now;

        public NetworkDescriptor(string trainDataPath, string testDataPath, string checkpointPath, NetworkType type, 
            int inputDimension, int outputClasses)
        {
            TrainDataPath = trainDataPath;
            TestDataPath = testDataPath;
            CheckpointPath = checkpointPath;
            Type = type;
            InputDimension = inputDimension;
            OutputClasses = outputClasses;
        }

        public int Epochs { get; set; } = 5000;
        public uint BatchSize { get; set; } = 100;
        public bool Evaluate { get; set; } = true;
        public int EpochCheckpoint { get; set; } = 10;
        public string FeaturesStreamName { get; set; } = "features";
        public string LabelsStreamName { get; set; } = "labels";
        public float LearningRatePerSample { get; set; } = 0.00125f;
        public string TrainDataPath { get; }
        public string TestDataPath { get; }
        private string CheckpointPath { get; }
        public NetworkType Type { get; }
        public int InputDimension { get; }
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