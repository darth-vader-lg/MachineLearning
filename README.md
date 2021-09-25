

# MachineLearning

**MachineLearning** is a .NET library mainly based on Microsoft [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) framework but it could be considered a melting pot for various frameworks ([TensorFlow](https://www.tensorflow.org/), [TensorFlow models](https://github.com/tensorflow/models), [PyTorch](https://pytorch.org/), [Ultralytics](https://ultralytics.com/) [YoloV5](https://github.com/ultralytics/yolov5), etc...).<BR>

## Main characteristics
* It's based on the [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) framework.
* Compatible with TensorFlow 2.6.x and [TensorFlow 2.6.x](https://www.tensorflow.org/) and [ONNX](https://onnx.ai/)
* All code can be written in any .NET standard languages (C#, F#, Basic, etc...) without knowledge or needs of resources as Python or anything other.
* It can be used with all .NET languages, simply including the package on your project.
* It has a multitasking structure, providing base classes which allow background models' train and update while using them for the inference without stopping; all in the same device.
* A growing model zoo with simple to use, ready-made and parametrized classes to solve main machine learning tasks.
* It includes obviously all the [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) features at low level, but also wrapping some of them with more friendly classes for newbies.
* An object detection class ([ObjectDetection.cs](https://github.com/darth-vader-lg/MachineLearning/blob/master/src/MachineLearning/ModelZoo/ObjectDetection.cs)), having both train and inference skills, is provided to bridge the gap of the missing train feature task of the .NET projects, which nowadays is accomplished mainly in Python.
* Can import a plenty of pre-trained models (TensorFlow saved_model or frozen graph, Onnx, etc...)

## Some ready-made classes
|Class|Purpose|
|--|--|
|[ImageClassification](https://github.com/darth-vader-lg/MachineLearning/blob/master/src/MachineLearning/ModelZoo/ImageClassification.cs)|Classify images in categories, from standard pre-trained models or training custom models|
|[ObjectDetection](https://github.com/darth-vader-lg/MachineLearning/blob/master/src/MachineLearning/ModelZoo/ObjectDetection.cs)|Detect objects in pictures with common standard pre-trained models or training custom models.|
|[SentenceClassification](https://github.com/darth-vader-lg/MachineLearning/blob/master/src/MachineLearning/ModelZoo/SentenceClassification.cs)|To classify the meaning of text/phrases.|
|[SizeEstimation](https://github.com/darth-vader-lg/MachineLearning/blob/master/src/MachineLearning/ModelZoo/SizeEstimation.cs)|Estimation of sizes from set of measures|
|[SmartDictionary](https://github.com/darth-vader-lg/MachineLearning/blob/master/src/MachineLearning/ModelZoo/SmartDictionary.cs)|A dictionary having string keys but with elements addressable with expressions having just a similarity to the requested key.|

## Getting started with MachineLearning

Simply include the package (or the reference to the project if you include it in your solution) to used the library.<BR>
Include extra packages and runtimes if you need to use more advanced features.<BR>
### Packages for advanced feature tasks:
* **Onnx models inference**: [Onnx runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) or [Onnx runtime GPU](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)
* **TensorFlow model inference**: [LG.TensorFlow.NET](https://www.nuget.org/packages/LG.TensorFlow.NET), [LG.SciSharp.TensorFlow.Redist](https://www.nuget.org/packages/LG.SciSharp.TensorFlow.Redist) or [LG.SciSharp.TensorFlow.Redist-Windows-GPU](https://www.nuget.org/packages/LG.SciSharp.TensorFlow.Redist-Windows-GPU).
* **TensorFlow object detection train**: [ODModelBuilderTF](https://www.nuget.org/packages/ODModelBuilderTF), [ODModelBuilderTF-Redist-Win](https://www.nuget.org/packages/ODModelBuilderTF-Redist-Win), [ODModelBuilderTF-Redist-Win-TF](https://www.nuget.org/packages/ODModelBuilderTF-Redist-Win-TF)
* **Pytorch Yolo v5 train**: *work in progress...*

## Examples
There is a an [examples directory](https://github.com/darth-vader-lg/MachineLearning/tree/master/examples) containing some simple demos.
For a more exhaustive usage cases it would be interesting to take a look in the [test set directory](https://github.com/darth-vader-lg/MachineLearning/tree/master/test) containing hundred of snippets and real application code of many models.

### Below some code example snippets:
**Object detection inference**
```C#
static void Main(string[] args)
{
   // Define the data
   var modelFile = ExampleData.File(
      root: "Workspace",
      path: Path.Combine("ssd_mobilenet_v2_320x320_coco17_tpu-8", "saved_model", "saved_model.pb"),
      url: "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz");
   var imageFile = ExampleData.File(
      root: "Workspace",
      path: "banana.jpg",
      url: "https://github.com/darth-vader-lg/ML-NET/raw/056c60479304a3b5dbdf129c9bc6e853322bb090/test/data/images/banana.jpg");

   // Download the data
   Console.WriteLine($"Downloading the model...");
   var modelPath = modelFile.Get();
   Console.WriteLine($"Downloading the image...");
   var imagePath = imageFile.Get();

   // Import the model
   using var m = new ObjectDetection { ModelStorage = new ModelStorageMemory { ImportPath = modelPath } };

   // Do predictions
   var dets = m.GetPrediction(imagePath);

   // Get the boxes, draw them on bitmap and save the marked bitmap
   var boxes = dets.GetBoxes(minScore: dets.DetectionScores.Max() * 0.8);
   using (var bmp = new Bitmap(Image.FromFile(imagePath))) {
      // Draw the boxes
      foreach (var box in boxes)
         DrawBoxesOnBitmap(bmp, box);
      // Save the marked image
      var dest = Path.ChangeExtension(imagePath, null) + ".scored" + Path.GetExtension(imagePath);
      bmp.Save(dest);
      // Print the results
      Console.WriteLine($"Found {boxes.Count} objects");
      foreach (var box in boxes)
         Console.WriteLine($"{box.Name} (id:{box.Id}) {box.Score * 100f:###.#}%");
      Console.WriteLine($"The image has been saved in {dest}");
   }
}
```
**Image classification train**
```C#
static void Main(string[] args)
{
   var euroSATImages = ExampleData.Folder(
      Path.Combine("Workspace"),
      "assets",
      "https://github.com/dotnet/machinelearning-samples/raw/04076c5f95814a735dd5ecdb17fcb2052b3c3c45/samples/modelbuilder/ImageClassification_Azure_LandUse/assets.zip");
         
   // Parameters
   var numCategories = 5;
   var trainImagesPerCategory = 20;
   var testImagesPerCategory = 5;
   var crossValidationFolds = 5;
         
   // Prepare the train folder with just a subset of the downloaded images
   var trainImagesFolder = Path.Combine("Workspace", "TrainImages");
   if (Directory.Exists(trainImagesFolder))
      Directory.Delete(trainImagesFolder, true);

   // Take a random subset of the images
   Console.WriteLine("Preparing the train environment...");
   var rnd = new Random(0);
   var folders = Directory.GetDirectories(euroSATImages.Get()).OrderBy(f => rnd.Next()).ToArray();
   var categories = new string[numCategories][];
   for (var i = 0; i < numCategories; i++) {
      categories[i] = Directory.GetFiles(folders[i], "*.jpg").OrderBy(f => rnd.Next()).Take(trainImagesPerCategory + testImagesPerCategory).ToArray();
      var dest = Path.Combine(trainImagesFolder, Path.GetFileName(folders[i]));
      Directory.CreateDirectory(dest);
      foreach (var image in categories[i].Take(trainImagesPerCategory))
         File.Copy(image, Path.Combine(dest, Path.GetFileName(image)), true);
   }

   // Create the model
   var model = new ImageClassification
   {
      DataStorage = new DataStorageBinaryMemory(),
      ImagesSources = new[] { trainImagesFolder },
      ModelStorage = new ModelStorageMemory(),
      ModelTrainer = new ModelTrainerCrossValidation { NumFolds = crossValidationFolds },
      Name = "Custom train"
   };

   // Log the messages
   MachineLearningContext.Default.Log += (sender, e) =>
   {
      // Filter trace messages but not about training phase 
      if (e.Kind < MachineLearningLogKind.Info && !e.Message.Contains("Phase: Bottleneck Computation") && !e.Message.Contains("Phase: Training"))
         return;
      Console.WriteLine(e.Message);
   };
                                              
   // Do predictions
   var predictions = (from category in categories
                        from file in category.Skip(trainImagesPerCategory).Take(testImagesPerCategory)
                        select (File: file, Result: model.GetPrediction(file))).ToArray();

   // Check predictions comparing the kind with the folder name containing the image
   var wrongPrediction = predictions.Where(prediction => string.Compare(prediction.Result.Kind, Path.GetFileName(Path.GetDirectoryName(prediction.File)), true) != 0);
   var rightPredictionPercentage = ((double)predictions.Length - wrongPrediction.Count()) * 100 / predictions.Length;
   if (wrongPrediction.Count() > 0) {
      Console.WriteLine("Wrong predictions:");
      foreach (var prediction in wrongPrediction)
         Console.WriteLine($"Expected {Path.GetFileName(Path.GetDirectoryName(prediction.File))} for {Path.GetFileName(prediction.File)}, got {prediction.Result.Kind}");
   }
   Console.WriteLine($"Right results percentage: {rightPredictionPercentage:###.#}%");
}
```
**Smart dictionary**
```C#
static void Main(string[] args)
{
   // Create the dictionary
   var dictionary = new SmartDictionary<string>()
   {
      { "this is a house", "house" },
      { "this is a car", "car" },
      { "this is a window", "window" },
   };

   // Test set of keys
   var similarKeys = new[]
   {
      "these are houses",
      "I see a car",
      "It seems a broken window"
   };

   // Query the dictionary
   foreach (var key in similarKeys)
      Console.WriteLine($"dictionary[\"{key}\"] => {dictionary.Similar[key]}");
}
```


## Packages
[LG.MachineLearning](https://www.nuget.org/packages/LG.MachineLearning): the machine learning library.

## License

ML.NET is licensed under the [MIT license](LICENSE) and it is free to use commercially.


