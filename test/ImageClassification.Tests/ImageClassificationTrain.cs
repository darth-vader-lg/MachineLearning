using ImageClassification.Tests;
using MachineLearning.Data;
using MachineLearning.Model;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ImageClassification")]
   public class ImageClassificationTrain : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ImageClassificationTrain(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test for image classification with cross evaluation
      /// </summary>
      [Fact, Trait("Category", "Train")]
      public void Train()
      {
         // Parameters
         var numCategories = 5;
         var trainImagesPerCategory = 20;
         var testImagesPerCategory = 5;
         var crossValidationFolds = 5;
         // Copy in the train folder a subset of train images just for test speed reason
         var trainImagesFolder = Path.Combine(DataFolder, "Land", "TrainImages");
         if (Directory.Exists(trainImagesFolder))
            Directory.Delete(trainImagesFolder, true);
         var rnd = new Random(0);
         var folders = Directory.GetDirectories(GetImagesFolder("EuroSAT").Get()).OrderBy(f => rnd.Next()).ToArray();
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
            if (e.Source != model.Name || e.Kind == MachineLearningLogKind.Trace) {
               Debug.WriteLine(e.Message);
               return;
            }
            WriteLine(e.Message);
         };
         // Do predictions
         var predictions = (from category in categories
                              from file in category.Skip(trainImagesPerCategory).Take(testImagesPerCategory)
                              select (File: file, Result: model.GetPrediction(file))).ToArray();
         // Check predictions comparing the kind with the folder name containing the image
         var wrongPrediction = predictions.Where(prediction => string.Compare(prediction.Result.Kind, Path.GetFileName(Path.GetDirectoryName(prediction.File)), true) != 0);
         var rightPredictionPercentage = ((double)predictions.Length - wrongPrediction.Count()) * 100 / predictions.Length;
         if (wrongPrediction.Count() > 0) {
            WriteLine("Wrong predictions:");
            foreach (var prediction in wrongPrediction)
               WriteLine($"Expected {Path.GetFileName(Path.GetDirectoryName(prediction.File))} for {Path.GetFileName(prediction.File)}, got {prediction.Result.Kind}");
         }
         WriteLine($"Right results percentage: {rightPredictionPercentage:###.#}%");
         Assert.True(rightPredictionPercentage > 90);
      }
      #endregion
   }
}
