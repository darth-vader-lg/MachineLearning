using Common.Tests;
using System;
using System.IO;
using System.Linq;
using Xunit.Abstractions;

namespace ModelConfig.Tests
{
   /// <summary>
   /// Environment for the object detection tests
   /// </summary>
   public class TestEnv : BaseEnvironment
   {
      #region Properties
      /// <summary>
      /// Image classification models
      /// </summary>
      public static TestData[] ImageClassificationModels { get; } = (from m in ImageClassification.Tests.TestEnv.Models
                                                                     where new[] { ".pb", ".onnx" }.Any(ext => ext == Path.GetExtension(m.FullPath).ToLower())
                                                                     select m).ToArray();
      /// <summary>
      /// Object detection models
      /// </summary>
      public static TestData[] ObjectDetectionModels { get; } = (from m in ObjectDetection.Tests.TestEnv.Models
                                                                 where new[] { ".pb", ".onnx", ".pt" }.Any(ext => ext == Path.GetExtension(m.FullPath).ToLower())
                                                                 select m).ToArray();
      /// <summary>
      /// All pretrained models
      /// </summary>
      public static TestData[] Models => (from g in new[] { ImageClassificationModels, ObjectDetectionModels }
                                          from m in g
                                          select m).ToArray();
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
      /// Return a known model
      /// </summary>
      /// <param name="name">Name of the model</param>
      /// <returns>The model or null</returns>
      public static TestData GetModel(string name) => Models.FirstOrDefault(model => model.Name == name);
      #endregion
   }
}
