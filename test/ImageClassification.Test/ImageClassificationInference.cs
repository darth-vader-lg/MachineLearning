using ImageRecognition.Test;
using MachineLearning.Model;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ImageRecognition")]
   public class ImageClassificationInference : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ImageClassificationInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test for image classification with cross evaluation
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void InferenceSingleThread()
      {
         // Create the model
         var model = new ImageClassification
         {
            ModelStorage = new ModelStorageFile(GetModel("Land").Path),
         };
         // Do predictions
         var predictions = 
            Directory.GetFiles(GetImagesFolder("Land inference").Path, "*.jpg", SearchOption.AllDirectories)
            .Select(file => (File: file, Result: model.GetPrediction(file)))
            .ToArray();
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
