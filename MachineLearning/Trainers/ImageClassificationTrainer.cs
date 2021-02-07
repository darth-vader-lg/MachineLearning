﻿using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Vision.ImageClassificationModelParameters;
using TTransformer = Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Vision.ImageClassificationModelParameters>;
using TTrainer = Microsoft.ML.Vision.ImageClassificationTrainer;
using TOptions = Microsoft.ML.Vision.ImageClassificationTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class ImageClassificationTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal ImageClassificationTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.ImageClassification(Options);
      #endregion
   }
}
