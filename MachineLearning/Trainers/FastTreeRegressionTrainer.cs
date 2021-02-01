﻿using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.FastTree.FastTreeRegressionModelParameters;
using MSTrainer = Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastTreeRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastTreeRegressionTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastTreeRegressionTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.FastTree(Options);
      #endregion
   }
}