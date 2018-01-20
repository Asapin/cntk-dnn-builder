using CNTK;

namespace NeuralNetwork
{
    public abstract class Activation
    {
        public delegate Function Apply(Function function);

        public static readonly Apply None = function => function;
        public static readonly Apply ReLU = function => CNTKLib.ReLU(function);
        public static readonly Apply LeakyReLU = function => CNTKLib.LeakyReLU(function);
        public static readonly Apply ELU = function => CNTKLib.ELU(function);
        public static readonly Apply Tanh = function => CNTKLib.Tanh(function);
        public static readonly Apply Sigmoid = function => CNTKLib.Sigmoid(function);
    }
}