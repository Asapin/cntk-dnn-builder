using CNTK;

namespace NeuralNetwork.Network.LearningRates
{
    public class StaticLearningRate : ILearningRate
    {
        private readonly float _learningRatePerSample;

        public StaticLearningRate(float learningRatePerSample)
        {
            _learningRatePerSample = learningRatePerSample;
        }

        public TrainingParameterScheduleDouble GetSchedule()
        {
            return new TrainingParameterScheduleDouble(_learningRatePerSample);
        }
    }
}