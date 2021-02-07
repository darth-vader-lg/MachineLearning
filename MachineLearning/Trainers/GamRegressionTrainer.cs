using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.FastTree.GamRegressionModelParameters;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.GamRegressionModelParameters>;
using TTrainer = Microsoft.ML.Trainers.FastTree.GamRegressionTrainer;
using TOptions = Microsoft.ML.Trainers.FastTree.GamRegressionTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe GamRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class GamRegressionTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
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
