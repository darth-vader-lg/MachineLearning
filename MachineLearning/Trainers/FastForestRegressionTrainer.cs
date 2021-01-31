using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.FastTree.FastForestRegressionModelParameters;
using MSTrainer = Microsoft.ML.Trainers.FastTree.FastForestRegressionTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastForestRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastForestRegressionTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastForestRegressionTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.FastForest(Options);
      #endregion
   }
}
