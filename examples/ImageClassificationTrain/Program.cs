using Common.Examples;
using MachineLearning;
using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.ModelZoo;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace ImageClassificationTrain
{
   class Program
   {
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
   }
}
