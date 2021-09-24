using MachineLearning.Data;
using MachineLearning.Model;
using ObjectDetection.Tests;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ObjectDetection")]
   public class ObjectDetectionTrain : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ObjectDetectionTrain(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Model train with tensorflow
      /// </summary>
      /// <param name="modelPath">Path of the model to import</param>
      [Fact, Trait("Category", "Train")]
      public void Train()
      {
         // Ensure empty environment to start the train from scratch
         var trainFolder = Path.GetFullPath(Path.Combine(DataFolder, "Train"));
         if (Directory.Exists(trainFolder))
            Directory.Delete(trainFolder, true);
         // Create the context with fixed seed
         using var m = new ObjectDetection()
         {
            BatchSize = 12,
            DataStorage = new DataStorageBinaryMemory(),
            MaxTrainingCycles = 200,
            MaxEvalTotalLoss = double.MaxValue, // So high just for test. Standard values are less than 1
            MaxStepTotalLoss = 1.5, // So high just for test. Standard values are less than 0.5
            MinEvalAveragePrecision = 0.3, // So low just for test. Standard value must be at least 0.6 / 0.7
            ModelStorage = new ModelStorageMemory(),
            Name = "Custom train",
            TrainFolder = trainFolder
         };
         // Log the messages
         MachineLearningContext.Default.Log += (sender, e) =>
         {
            if (e.Kind == MachineLearningLogKind.Trace)
               return;
            else if (e.Source != m.Name) {
               Debug.WriteLine(e.Message);
               return;
            }
            WriteLine(e.Message);
         };
         // Define the sets of images
         m.UpdateStorageByFolders(new[] { GetImagesFolder("Carps train").Get() }, new[] { GetImagesFolder("Carps eval").Get() });
         // Start train and wait for max 30 minutes
         m.StartTraining();
         Assert.True(Task.Run(() => m.WaitModelChanged()).Wait(new TimeSpan(0, 30, 0)), "Train timeout");
         // Check if the train generated the checkpoints and exported the model.onnx
         Assert.True(File.Exists(Path.Combine(trainFolder, "train", "checkpoint")));
         Assert.True(Directory.Exists(Path.Combine(trainFolder, "export", "saved_model")));
         Assert.True(File.Exists(Path.Combine(trainFolder, "export", "model.onnx")));
         // Test the inference
         foreach (var image in Directory.GetFiles(GetImagesFolder("Carps eval").Get(), "*.jpg")) {
            var boxes = m.GetPrediction(image).GetBoxes(0.9);
            Assert.True(boxes.Count > 0);
            WriteLine($"{Path.GetFileName(image)}: boxes {boxes.Count}, score {boxes.Max(b => b.Score) * 100:###.#}%");
         }
      }
      #endregion
   }
}
