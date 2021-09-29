using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace ModelConfig.Tests
{
   public class ModelConfigInference : TestEnv
   {
      #region class ImageClassificationSet
      /// <summary>
      /// Image classification models
      /// </summary>
      internal class ImageClassificationSet : IEnumerable<object[]>
      {
         public IEnumerator<object[]> GetEnumerator()
         {
            foreach (var model in ImageClassificationModels)
               yield return new object[] { model.Name };
         }
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
      }
      #endregion
      #region class ObjectDetectionSet
      /// <summary>
      /// Image classification models
      /// </summary>
      internal class ObjectDetectionSet : IEnumerable<object[]>
      {
         public IEnumerator<object[]> GetEnumerator()
         {
            foreach (var model in ObjectDetectionModels)
               yield return new object[] { model.Name };
         }
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
      }
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ModelConfigInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test automatic inference of image classification model type
      /// </summary>
      [Theory, ClassData(typeof(ImageClassificationSet)), Trait("Category", "Configuration")]
      public void ImageClassification(string model) => InferConfiguration(model);
      /// <summary>
      /// Test automatic inference of object detection model type
      /// </summary>
      [Theory, ClassData(typeof(ObjectDetectionSet)), Trait("Category", "Configuration")]
      public void ObjectDetection(string model) => InferConfiguration(model);
      /// <summary>
      /// Infer the configuration of a model
      /// </summary>
      private void InferConfiguration(string model)
      {
         var modelFile = GetModel(model);
         var expectedData = File.ReadAllLines("ModelConfigs.txt");
         var sbErrors = new StringBuilder();
         // Build info
         var sb = new StringBuilder();
         var modelHeader = $"Model: {model}";
         sb.AppendLine(modelHeader);
         sb.AppendLine($"File: {Path.GetFileName(modelFile.Get())}");
         using var cfg = MachineLearning.Model.ModelConfig.Load(modelFile.Get());
         sb.AppendLine($"Model format: {cfg.Format}");
         sb.AppendLine($"Model type: {cfg.ModelType}");
         sb.AppendLine($"Model category: {cfg.ModelCategory}");
         sb.AppendLine($"Image size: {(cfg.ImageSize.Width > -1 ? cfg.ImageSize.Width : "?")}x{(cfg.ImageSize.Height > -1 ? cfg.ImageSize.Height : "?")}");
         if (cfg.OffsetImage != 0f)
            sb.AppendLine($"Offset image: {cfg.OffsetImage.ToString(CultureInfo.InvariantCulture)}");
         if (cfg.ScaleImage != 1f)
            sb.AppendLine($"Scale image: {cfg.ScaleImage.ToString(CultureInfo.InvariantCulture)}");
         if (cfg.Model.ToString() is string text && !string.IsNullOrEmpty(text)) {
            sb.AppendLine("Model info:");
            sb.AppendLine(cfg.Model.ToString());
         }
         if (cfg.Inputs.Count > 0) {
            sb.AppendLine("Inputs:");
            foreach (var tensor in cfg.Inputs)
               sb.AppendLine(tensor.ToString());
         }
         if (cfg.Outputs.Count > 0) {
            sb.AppendLine("Outputs:");
            foreach (var tensor in cfg.Outputs)
               sb.AppendLine(tensor.ToString());
         }
         if (cfg.Inputs.Count > 0 && cfg.Outputs.Count > 0) {
            sb.AppendLine("Column names:");
            foreach (var c in Enum.GetNames<MachineLearning.Model.ModelConfig.ColumnTypes>()) {
               try {
                  sb.AppendLine($"{c} column name: {cfg.GetColumnName(Enum.Parse<MachineLearning.Model.ModelConfig.ColumnTypes>(c))}");
               }
               catch (Exception) {
               }
            }
         }
         if (cfg.Labels.Count > 0) {
            sb.AppendLine("Labels:");
            foreach (var chunk in from l in cfg.Labels.Select((label, index) => (label, index))
                                    group l by l.index / 10 into g
                                    select string.Join('|', from item in g select item.label))
               sb.AppendLine(chunk);
         }
         sb.Append("==============================================");
         WriteLine(sb.ToString());
         // Compare with expected info
         var expectedEnum = expectedData.AsEnumerable().GetEnumerator();
         var resultEnum = sb.ToString().Split(Environment.NewLine).AsEnumerable().GetEnumerator();
         resultEnum.MoveNext();
         var found = false;
         while (expectedEnum.MoveNext()) {
            if (expectedEnum.Current == resultEnum.Current) {
               found = true;
               var sbCurrentError = new StringBuilder();
               while (expectedEnum.MoveNext() && resultEnum.MoveNext()) {
                  if (expectedEnum.Current.StartsWith("======") || resultEnum.Current.StartsWith("====="))
                     break;
                  if (expectedEnum.Current != resultEnum.Current) {
                     if (sbCurrentError.Length == 0)
                        sbCurrentError.AppendLine(modelHeader);
                     sbCurrentError.AppendLine($"Expected: {expectedEnum.Current}; got {resultEnum.Current}");
                     break;
                  }
               }
               if (sbCurrentError.Length > 0)
                  sbErrors.AppendLine(sbCurrentError.ToString());
               break;
            }
         }
         if (!found) {
            sbErrors.AppendLine($"Model {model} not found in the expected data file ModelConfig.txt.");
            sbErrors.AppendLine("Cannot check result.");
         }
         Assert.True(sbErrors.Length == 0, $"Errors!!!{Environment.NewLine}{sbErrors}");
      }
      #endregion
   }
}
