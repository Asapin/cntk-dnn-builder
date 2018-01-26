using System;
using System.IO;
using System.Linq;
using System.Net;

namespace NeuralNetwork.Utils
{
    public static class DownloadUtils
    {
        
        private const string TestDataUrl = "https://pjreddie.com/media/files/mnist_test.csv"; 
        private const string TrainDataUrl = "https://pjreddie.com/media/files/mnist_train.csv";

        private const string TestDataFileLocation = "../../Datasets/mnist/test.csv";
        private const string TrainDataFileLocation = "../../Datasets/mnist/train.csv";

        public static void DownloadMnist()
        {
            if (!File.Exists(TestDataFileLocation))
            {
                Console.WriteLine("Downloading test data...");
                DownloadFile(TestDataUrl, TestDataFileLocation);
                Console.WriteLine("Downloading finished.");
                Console.WriteLine("Reformating test data...");
                ReformatFile(TestDataFileLocation);
                Console.WriteLine("Reformating finished.");
            }

            if (!File.Exists(TrainDataFileLocation))
            {
                Console.WriteLine("Downloading train data...");
                DownloadFile(TrainDataUrl, TrainDataFileLocation);
                Console.WriteLine("Downloading finished.");
                Console.WriteLine("Reformating train data...");
                ReformatFile(TrainDataFileLocation);
                Console.WriteLine("Reformating finished.");
                
            }
        }

        private static void DownloadFile(string url, string saveLocation)
        {
            var directory = Path.GetDirectoryName(saveLocation);
            Directory.CreateDirectory(directory ?? throw new FileNotFoundException($"Couldn't create file {saveLocation}"));

            using (var client = new WebClient())
            {
                client.DownloadFile(url, saveLocation);
            }
        }

        private static void ReformatFile(string fileLocation)
        {
            var formatedData = File.ReadAllLines(fileLocation).Select(line =>
            {
                var columns = line.Split(',');
                var onehot = new int[10];

                onehot[int.Parse(columns[0])] = 1;
                return "|labels " + onehot.Aggregate("", (s, i) => s + i + " ") + "|features " +
                       columns.Skip(1).Aggregate("", (s, s1) => s + s1 + " ");
            });
            
            File.WriteAllLines(fileLocation, formatedData);
        }
    }
}