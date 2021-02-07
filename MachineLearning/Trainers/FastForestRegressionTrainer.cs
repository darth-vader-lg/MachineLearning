using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.FastTree.FastForestRegressionModelParameters;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastForestRegressionModelParameters>;
using TTrainer = Microsoft.ML.Trainers.FastTree.FastForestRegressionTrainer;
using TOptions = Microsoft.ML.Trainers.FastTree.FastForestRegressionTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastForestRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastForestRegressionTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastForestRegressionTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.FastForest(Options);
      #endregion
   }
}
