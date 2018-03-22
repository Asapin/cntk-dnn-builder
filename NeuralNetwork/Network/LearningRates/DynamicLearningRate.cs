using System.Collections.Generic;
using CNTK;

namespace NeuralNetwork.Network.LearningRates
{
    public struct DynamicRate
    {
        public readonly uint Multyplier;
        public readonly double Rate;

        public DynamicRate(uint multyplier, double rate)
        {
            Multyplier = multyplier;
            Rate = rate;
        }
    }

    public class DynamicLearningRate : ILearningRate
    {
        private readonly uint _trainingMinibatchesCount;
        private readonly IEnumerable<DynamicRate> _dynamicRates;

        public DynamicLearningRate(uint trainingMinibatchesCount, IEnumerable<DynamicRate> dynamicRates)
        {
            _trainingMinibatchesCount = trainingMinibatchesCount;
            _dynamicRates = dynamicRates;
        }

        public TrainingParameterScheduleDouble GetSchedule()
        {
            var vector = new VectorPairSizeTDouble();
            foreach (var pair in _dynamicRates)
            {
                var rate = new PairSizeTDouble(pair.Multyplier, pair.Rate);
                vector.Add(rate);
            }
            return new TrainingParameterScheduleDouble(vector, _trainingMinibatchesCount);
        }
    }
}