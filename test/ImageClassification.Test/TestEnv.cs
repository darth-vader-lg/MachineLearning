using Common.Tests;
using MachineLearning;
using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.ModelZoo;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ImageRecognition.Test
{
   /// <summary>
   /// Environment for the image classification tests
   /// </summary>
   public class TestEnv : BaseEnvironment
   {
      #region Properties
      /// <summary>
      /// Image folders
      /// </summary>
      public static TestData[] ImageFolders { get; } = new[]
      {
         TestData.Folder(
            "EuroSAT",
            Path.Combine("Data", "EuroSAT", "Images"),
            "assets",
            "https://github.com/dotnet/machinelearning-samples/raw/04076c5f95814a735dd5ecdb17fcb2052b3c3c45/samples/modelbuilder/ImageClassification_Azure_LandUse/assets.zip"),
         TestData.Folder(
            "Land train",
            Path.Combine("Data", "Land"),
            "TrainImages",
            BuildLandImages),
         TestData.Folder(
            "Land inference",
            Path.Combine("Data", "Land"),
            "InferenceImages",
            BuildLandImages),
      };
      /// <summary>
      /// Pretrained models
      /// </summary>
      public static TestData[] Models { get; } = new[]
      {
         TestData.File(
            "Land",
            Path.Combine("Data", "Land"),
            Path.Combine("Model.zip"),
            BuildLandModel)
      };
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      internal TestEnv(ITestOutputHelper output = null) : base(output)
      {
         // Disable cuda for tests
         Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "-1");
      }
      /// <summary>
      /// Create the Land model images from a subset of EuroSAT images
      /// </summary>
      /// <param name="fullPath">Full path of the object to create</param>
      private static void BuildLandImages(string fullPath)
      {
         // Parameters
         var numCategories = 3;
         var trainImagesPerCategory = 20;
         var inferenceImagesPerCategory = 5;
         // Copy in the train folder a subset of train images just for test speed reason
         var trainImagesFolder = Path.Combine("Data", "Land", "TrainImages");
         var inferenceImagesFolder = Path.Combine("Data", "Land", "InferenceImages");
         if (Directory.Exists(trainImagesFolder))
            Directory.Delete(trainImagesFolder, true);
         if (Directory.Exists(inferenceImagesFolder))
            Directory.Delete(inferenceImagesFolder, true);
         var rnd = new Random(0);
         var folders = Directory.GetDirectories(GetImagesFolder("EuroSAT").Path).OrderBy(f => rnd.Next()).ToArray();
         var categories = new string[numCategories][];
         for (var i = 0; i < numCategories; i++) {
            categories[i] = Directory.GetFiles(folders[i], "*.jpg").OrderBy(f => rnd.Next()).Take(trainImagesPerCategory + inferenceImagesPerCategory).ToArray();
            var dest = Path.Combine(trainImagesFolder, Path.GetFileName(folders[i]));
            Directory.CreateDirectory(dest);
            foreach (var image in categories[i].Take(trainImagesPerCategory))
               File.Copy(image, Path.Combine(dest, Path.GetFileName(image)), true);
            dest = Path.Combine(inferenceImagesFolder, Path.GetFileName(folders[i]));
            Directory.CreateDirectory(dest);
            foreach (var image in categories[i].Skip(trainImagesPerCategory).Take(inferenceImagesPerCategory))
               File.Copy(image, Path.Combine(dest, Path.GetFileName(image)), true);
         }
      }
      /// <summary>
      /// Create the Land model from EuroSAT images
      /// </summary>
      /// <param name="fullPath">Full path of the object to create</param>
      private static void BuildLandModel(string fullPath)
      {
         // Create the model
         using var context = new MachineLearningContext(0);
         using var model = new ImageClassification(context)
         {
            DataStorage = new DataStorageBinaryMemory(),
            ImagesSources = new[] { GetImagesFolder("Land train").Path },
            ModelStorage = new ModelStorageFile(fullPath),
            ModelTrainer = new ModelTrainerCrossValidation { NumFolds = 5 },
         };
         // Log the messages
         context.Log += (sender, e) => //@@@
         {
            // Filter trace messages but not about training phase 
            if (e.Kind < MachineLearningLogKind.Info && !e.Message.Contains("Phase: Bottleneck Computation") && !e.Message.Contains("Phase: Training"))
               return;
            Debug.WriteLine(e.Message);
         };
         // Start the train
         model.StartTraining();
         // Wait for completion
         model.WaitTrainingEnded();
         // Check the model existence
         Assert.True(File.Exists(fullPath));
      }
      /// <summary>
      /// Return a known model
      /// </summary>
      /// <param name="name">Name of the model</param>
      /// <returns>The model or null</returns>
      public static TestData GetModel(string name) => Models.FirstOrDefault(model => model.Name == name);
      /// <summary>
      /// Return a known images folder
      /// </summary>
      /// <param name="name">Name of the images folder</param>
      /// <returns>The path of the images folder or null</returns>
      public static TestData GetImagesFolder(string name) => ImageFolders.FirstOrDefault(folder => folder.Name == name);
      #endregion
   }
}
