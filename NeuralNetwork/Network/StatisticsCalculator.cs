using CNTK;

namespace NeuralNetwork.Network
{
    public class StatisticsCalculator
    {
        private int _samples;
        private double _totalLoss;
        private double _totalEvaluation;

        public void LogStats(ref Trainer trainer)
        {
            _samples++;
            _totalLoss += trainer.PreviousMinibatchLossAverage();
            _totalEvaluation += trainer.PreviousMinibatchEvaluationAverage();
        }

        public void Reset()
        {
            _samples = 0;
            _totalLoss = 0;
            _totalEvaluation = 0;
        }

        public double GetAverageLoss()
        {
            return _totalLoss / _samples;
        }

        public double GetEvaluationAverage()
        {
            return _totalEvaluation / _samples;
        }
    }
}