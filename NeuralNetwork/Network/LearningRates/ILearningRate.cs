using CNTK;

namespace NeuralNetwork.Network.LearningRates
{
    public interface ILearningRate
    {
        TrainingParameterScheduleDouble GetSchedule();
    }
}