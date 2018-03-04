﻿using CNTK;

namespace NeuralNetwork.Layers
{
    public class DropoutLayer : AbstractLayer
    {
        private readonly double _dropoutRate;

        public DropoutLayer(double dropoutRate)
        {
            _dropoutRate = dropoutRate;
        }

        public override Function Layer(ref Function input, ref DeviceDescriptor device)
        {
            return CNTKLib.Dropout(input, _dropoutRate);
        }
    }
}