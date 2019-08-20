# Head Detection DCNN 
Deep convolutional neural network (DCNN) for head detection in Space Station.
The DCNN was deployed on TX2 with TensorRT 3.0 inference engine.

For the layers in the head detection network that cannot be supported directly by TensorRT, we implemented them with the user-defined layer interface PluginFactory.  

These layers are softmax layer(axis=2), concat layer(axis=2), reshape layer and flatten layer. 
