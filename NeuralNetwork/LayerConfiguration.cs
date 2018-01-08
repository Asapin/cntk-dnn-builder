namespace NeuralNetwork
{
    public class LayerConfiguration
    {
        public int NeuronCount;
        public Activation Activation;

        public LayerConfiguration(int neuronCount, Activation activation)
        {
            NeuronCount = neuronCount;
            Activation = activation;
        }
    }
}