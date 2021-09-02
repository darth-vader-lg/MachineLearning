using ImageRecognition.Test;
using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Serialization;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ImageRecognition")]
   public class ImageClassificationSerialization : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ImageClassificationSerialization(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test clone for image recognition model
      /// </summary>
      [Fact, Trait("Category", "Serialization")]
      public void Clone()
      {
         // Parameters
         var numCategories = 2;
         var trainImagesPerCategory = 20;
         var testImagesPerCategory = 5;
         // Copy in the train folder a subset of train images just for test speed reason
         var trainImagesFolder = Path.Combine("WorkSpace", "Land", "TrainImages");
         if (Directory.Exists(trainImagesFolder))
            Directory.Delete(trainImagesFolder, true);
         var rnd = new Random(0);
         var folders = Directory.GetDirectories(GetImagesFolder("EuroSAT").Path).OrderBy(f => rnd.Next()).ToArray();
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
            ModelTrainer = new ModelTrainerStandard(),
         };
         // Do predictions
         var predictions = (from category in categories
                            from file in category.Skip(trainImagesPerCategory).Take(testImagesPerCategory)
                            select model.GetPrediction(file)).ToArray();
         // Ensure the model is settled and all background task are completed
         model.StopTraining();
         // Clone the model
         var cloned = Serializer.Clone(model);
         model.Dispose();
         // Do predictions
         var clonedPredictions = (from category in categories
                                  from file in category.Skip(trainImagesPerCategory).Take(testImagesPerCategory)
                                  select cloned.GetPrediction(file)).ToArray();
         // Check predictions
         foreach (var prediction in predictions.Zip(clonedPredictions)) {
            Assert.Equal(prediction.First.Kind, prediction.Second.Kind, ignoreCase: true);
            Assert.Equal(prediction.First.Score, prediction.Second.Score, 3);
         }
      }
      #endregion
   }
}
