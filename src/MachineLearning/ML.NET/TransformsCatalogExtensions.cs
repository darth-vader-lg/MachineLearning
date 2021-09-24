using MachineLearning.Transforms;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Collections.Generic;
using System.Reflection;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni del catalogo di trasformazioni
   /// </summary>
   public static class TransformsCatalogExtensions
   {
      #region Methods
      /// <summary>
      /// Aggiunge una colonna con una costante
      /// </summary>
      /// <param name="catalog">Il catalogo di estensioni</param>
      /// <param name="outputColumnName">Nome della colonna contenente il valore costante</param>
      /// <param name="value">Il valore costante da aggiungere</param>
      /// <returns>L'Estimator</returns>
      /// <remarks>
      /// L'aggiunta della costante viene effettuata utilizzando l'Expression.
      /// Considerare il testo passato come facente parte della parte destra dell'assegnazione.
      /// </remarks>
      public static ConstantValueEstimator AddConst(this TransformsCatalog catalog, string outputColumnName, string value) => new(catalog, outputColumnName, value);
      /// <summary>
      /// Restituisce l'interfaccia IChannelProvider di un catalogo
      /// </summary>
      /// <param name="catalog"></param>
      /// <returns>L'interfaccia</returns>
      internal static IChannelProvider GetChannelProvider(this TransformsCatalog catalog)
      {
         var prop = catalog.GetType().GetProperty("Microsoft.ML.Data.IInternalCatalog.Environment", BindingFlags.Instance | BindingFlags.NonPublic);
         return prop.GetMethod.Invoke(catalog, null) as IChannelProvider;
      }
      /// <summary>
      /// Score the Inception image classification model
      /// </summary>
      /// <param name="catalog">Transforms catalog</param>
      /// <param name="inputColumnName">Name of the input column (the output of the Inception model)</param>
      /// <param name="predictedLabelColumnName">Name of the predicted label column</param>
      /// <param name="scoreColumnName">Name of the predicted scores column</param>
      /// <param name="labels">Set of the labels associate with the model</param>
      /// <returns>The estimator</returns>
      public static InceptionScorerEstimator ScoreInception(
         this TransformsCatalog catalog,
         string inputColumnName = InceptionScorerEstimator.Options.DefaultInputColumnName,
         string predictedLabelColumnName = DefaultColumnNames.PredictedLabel,
         string scoreColumnName = DefaultColumnNames.Score,
         IEnumerable<string> labels = null) =>
         ScoreInception(
            catalog,
            new InceptionScorerEstimator.Options
            {
               InputColumnName = inputColumnName,
               PredictedLabelColumnName = predictedLabelColumnName,
               ScoreColumnName = scoreColumnName,
               Labels = labels
            });
      /// <summary>
      /// Score the Inception image classification model
      /// </summary>
      /// <param name="catalog">Transforms catalog</param>
      /// <param name="options">Options</param>
      /// <returns>The estimator</returns>
      public static InceptionScorerEstimator ScoreInception(this TransformsCatalog catalog, InceptionScorerEstimator.Options options) => new(catalog.GetEnvironment(), options);
      /// <summary>
      /// Score a standard TensorFlow object detection model
      /// </summary>
      /// <param name="catalog">Transforms catalog</param>
      /// <param name="options">Options</param>
      /// <returns>The estimator</returns>
      public static TFStandardODScorerEstimator ScoreTensorFlowStandardObjectDetection(this TransformsCatalog catalog, TFStandardODScorerEstimator.Options options) => new(catalog.GetEnvironment(), options);
      /// <summary>
      /// Score a standard TensorFlow object detection model
      /// </summary>
      /// <param name="catalog">Transforms catalog</param>
      /// <param name="outputClassesColumnName">Output column with detection classes</param>
      /// <param name="outputScoresColumnName">Output column with detection scores</param>
      /// <param name="outputBoxesColumnName">Output column with detection boxes in format left, top, right, bottom from 0 to 1</param>
      /// <param name="outputLabelsColumnName">Output column with detection labels</param>
      /// <param name="inputClassesColumnName">Input column with detection classes</param>
      /// <param name="inputScoresColumnName">Input column with detection scores</param>
      /// <param name="inputBoxesColumnName">Input column with detection boxes in format left, top, right, bottom from 0 to 1</param>
      /// <param name="minScore">Minimum score for detection</param>
      /// <param name="labels">Set of the labels associate with the model</param>
      /// <returns>The estimator</returns>
      public static TFStandardODScorerEstimator ScoreTensorFlowStandardObjectDetection(
         this TransformsCatalog catalog,
         string outputClassesColumnName = TFStandardODScorerEstimator.Options.DefaultClassesColumnName,
         string outputScoresColumnName = TFStandardODScorerEstimator.Options.DefaultScoresColumnName,
         string outputBoxesColumnName = TFStandardODScorerEstimator.Options.DefaultBoxesColumnName,
         string outputLabelsColumnName = TFStandardODScorerEstimator.Options.DefaultLabelsColumnName,
         string inputClassesColumnName = TFStandardODScorerEstimator.Options.DefaultClassesColumnName,
         string inputScoresColumnName = TFStandardODScorerEstimator.Options.DefaultScoresColumnName,
         string inputBoxesColumnName = TFStandardODScorerEstimator.Options.DefaultBoxesColumnName,
         float minScore = TFStandardODScorerEstimator.Options.DefaultMinScore,
         IEnumerable<string> labels = null) => ScoreTensorFlowStandardObjectDetection(
            catalog,
            new TFStandardODScorerEstimator.Options
            {
               InputBoxesColumnName = inputBoxesColumnName,
               InputClassesColumnName = inputClassesColumnName,
               InputScoresColumnName = inputScoresColumnName,
               OutputClassesColumnName = outputClassesColumnName,
               OutputScoresColumnName = outputScoresColumnName,
               OutputBoxesColumnName = outputBoxesColumnName,
               OutputLabelsColumnName = outputLabelsColumnName,
               MinScore = minScore,
               Labels = labels
            });
      /// <summary>
      /// Score the Yolo V5 model
      /// </summary>
      /// <param name="catalog">Transforms catalog</param>
      /// <param name="options">Options</param>
      /// <returns>The estimator</returns>
      public static YoloV5ScorerEstimator ScoreYoloV5(this TransformsCatalog catalog, YoloV5ScorerEstimator.Options options) => new(catalog.GetEnvironment(), options);
      /// <summary>
      /// Score the Yolo V5 model
      /// </summary>
      /// <param name="catalog">Transforms catalog</param>
      /// <param name="outputClassesColumnName">Output column with detection classes</param>
      /// <param name="outputScoresColumnName">Output column with detection scores</param>
      /// <param name="outputBoxesColumnName">Output column with detection boxes in format left, top, right, bottom from 0 to 1</param>
      /// <param name="outputLabelsColumnName">Output column with detection labels</param>
      /// <param name="inputScoresColumnName">Name of the input column (the output of the Yolo model)</param>
      /// <param name="minScoreConfidence">Minimum score confidence for each single cell</param>
      /// <param name="minPerCategoryConfidence">Minimum score confidence for each category</param>
      /// <param name="imageWidth">The width of the model image</param>
      /// <param name="imageHeight">The height of the model image</param>
      /// <param name="labels">Set of the labels associate with the model</param>
      /// <returns>The estimator</returns>
      public static YoloV5ScorerEstimator ScoreYoloV5(
         this TransformsCatalog catalog,
         string outputClassesColumnName = YoloV5ScorerEstimator.Options.DefaultClassesColumnName,
         string outputScoresColumnName = YoloV5ScorerEstimator.Options.DefaultScoresColumnName,
         string outputBoxesColumnName = YoloV5ScorerEstimator.Options.DefaultBoxesColumnName,
         string outputLabelsColumnName = YoloV5ScorerEstimator.Options.DefaultLabelsColumnName,
         string inputScoresColumnName = YoloV5ScorerEstimator.Options.DefaultInputColumnName,
         float minScoreConfidence = YoloV5ScorerEstimator.Options.DefaultMinScoreConfidence,
         float minPerCategoryConfidence = YoloV5ScorerEstimator.Options.DefaultMinPerCategoryConfidence,
         int imageWidth = YoloV5ScorerEstimator.Options.DefaultImageWidth,
         int imageHeight = YoloV5ScorerEstimator.Options.DefaultImageHeight,
         IEnumerable<string> labels = null) => ScoreYoloV5(
            catalog,
            new YoloV5ScorerEstimator.Options
            {
               InputScoresColumnName = inputScoresColumnName,
               OutputClassesColumnName = outputClassesColumnName,
               OutputScoresColumnName = outputScoresColumnName,
               OutputBoxesColumnName = outputBoxesColumnName,
               OutputLabelsColumnName = outputLabelsColumnName,
               MinScoreConfidence = minScoreConfidence,
               MinPerCategoryConfidence = minPerCategoryConfidence,
               ImageWidth = imageWidth,
               ImageHeight = imageHeight,
               Labels = labels
            });
      #endregion
   }
}
