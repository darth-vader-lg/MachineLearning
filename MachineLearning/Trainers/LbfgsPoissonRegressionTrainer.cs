using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.PoissonRegressionModelParameters>;
using TTrainer = Microsoft.ML.Trainers.LbfgsPoissonRegressionTrainer;
using TOptions = Microsoft.ML.Trainers.LbfgsPoissonRegressionTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LbfgsPoissonRegressionTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LbfgsPoissonRegressionTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.LbfgsPoissonRegression(Options);
      #endregion
   }
}
