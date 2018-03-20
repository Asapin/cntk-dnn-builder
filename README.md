# Deep Neural Network Builder
This app allows you to build your own deep neural network by combining different layers without the need to implement them yourself.

# How to use
* [Install CNTK 2.4.0 on your machine](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)
* Create new class, that implements AbstractModel
* Override methods ``Before``, ``GetNetworkDescriptor``, ``GetLayers``
* Create new instance of your class, pass ``args[0]`` as a constructor parameter
* Run the app with a path to a folder where to store model's checlpoints as a comand line parameter

# Examples
You can find examples of different network models [here](https://github.com/Asapin/cntk-dnn-builder/tree/master/NeuralNetwork/PreparedModels)
