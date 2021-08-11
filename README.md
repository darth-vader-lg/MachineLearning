
# MachineLearning

**MachineLearning** is a .NET library mainly based on Microsoft [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) framework but it could be considered a melting pot of various frameworks ([TensorFlow](https://www.tensorflow.org/), [TensorFlow models](https://github.com/tensorflow/models), [PyTorch](https://pytorch.org/), [Ultralytics](https://ultralytics.com/) [YoloV5](https://github.com/ultralytics/yolov5), etc...).<BR>

## Main characteristics
* It's based on the [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) framework.<BR>
* All code can be written in any .NET standard languages (C#, F#, Basic, etc...) without knowledge or needs of resources as Python or anything other.
* It can be used with all .NET languages simply including the package on your project.<BR>
* It has a multitasking structure, providing base classes and which allow background models' train and update while using them for the inference without stopping; all in the same device.<BR>
* A growing model zoo with simple to use and parametrized classes to solve main machine learning tasks.<BR>
* It includes obviously all the [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) features at low level, but also wrapping some of them with more friendly classes for newbies.
* An object detection class ([ObjectDetection.cs](https://github.com/darth-vader-lg/MachineLearning/blob/master/ModelZoo/ObjectDetection.cs)), having both train and inference skills, is provided to bridge the gap of the missing train feature task of the .NET projects, which nowadays is accomplished mainly in Python.

## Getting started with MachineLearning

Simply include the package (or the reference to the project if you include it in your solution) to used the library.<BR>
Include extra packages and runtimes if you need to use more advanced features.<BR>
### Packages for advanced feature tasks:
* **Onnx models inference**: [Onnx runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) or [Onnx runtime GPU](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)
* **TensorFlow model inference**: [LG.TensorFlow.NET](https://www.nuget.org/packages/LG.TensorFlow.NET), [LG.SciSharp.TensorFlow.Redist](https://www.nuget.org/packages/LG.SciSharp.TensorFlow.Redist) or [LG.SciSharp.TensorFlow.Redist-Windows-GPU](https://www.nuget.org/packages/LG.SciSharp.TensorFlow.Redist-Windows-GPU).
* **TensorFlow object detection train**: [ODModelBuilderTF](https://www.nuget.org/packages/ODModelBuilderTF), [ODModelBuilderTF-Redist-Win](https://www.nuget.org/packages/ODModelBuilderTF-Redist-Win), [ODModelBuilderTF-Redist-Win-TF](https://www.nuget.org/packages/ODModelBuilderTF-Redist-Win-TF)
* **Pytorch Yolo v5 train**: *work in progress...*

## Packages
[LG.MachineLearning](https://www.nuget.org/packages/LG.MachineLearning): the machine learning library.<BR>

## License

ML.NET is licensed under the [MIT license](LICENSE) and it is free to use commercially.
