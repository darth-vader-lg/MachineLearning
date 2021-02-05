﻿using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers multiclasse
   /// </summary>
   public class MulticlassClassificationTrainersCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal MulticlassClassificationTrainersCatalog(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo ImageClassification
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public ImageClassificationTrainer ImageClassification(Microsoft.ML.Vision.ImageClassificationTrainer.Options options = default) =>
         new ImageClassificationTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LbfgsMaximumEntropyMulticlass
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LbfgsMaximumEntropyMulticlassTrainer LbfgsMaximumEntropy(Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer.Options options = default) =>
         new LbfgsMaximumEntropyMulticlassTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbmMulticlassTrainer
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LightGbmMulticlassTrainer LightGbm(Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer.Options options = default) =>
         new LightGbmMulticlassTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbmMulticlassTrainer
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public NaiveBayesMulticlassTrainer NaiveBayes(string labelColumnName = "Label", string featuresColumnName = "Features") =>
         new NaiveBayesMulticlassTrainer(this, (LabelColumnName: labelColumnName, FeaturesColumnName: featuresColumnName));
      /// <summary>
      /// Restituisce un trainer di tipo SdcaNonCalibratedMulticlass
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaNonCalibratedMulticlassTrainer SdcaNonCalibrated(Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options options = default) =>
         new SdcaNonCalibratedMulticlassTrainer(this, options);
      #endregion
   }
}
