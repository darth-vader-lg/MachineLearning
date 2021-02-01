using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.PoissonRegressionModelParameters;
using MSTrainer = Microsoft.ML.Trainers.LbfgsPoissonRegressionTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LbfgsPoissonRegressionTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LbfgsPoissonRegressionTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.LbfgsPoissonRegression(Options);
      #endregion
   }
}
