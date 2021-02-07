using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.LightGbm.LightGbmRegressionModelParameters>;
using TTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer;
using TOptions = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LightGbmRegressionTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LightGbmRegressionTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.LightGbm(Options);
      #endregion
   }
}
