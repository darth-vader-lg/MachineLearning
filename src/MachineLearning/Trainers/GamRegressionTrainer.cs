using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.FastTree.GamRegressionTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.FastTree.GamRegressionTrainer;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.GamRegressionModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe GamRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class GamRegressionTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal GamRegressionTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.Gam(Options);
      #endregion
   }
}
