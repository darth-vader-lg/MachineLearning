using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeRegressionModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastTreeRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastTreeRegressionTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastTreeRegressionTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.FastTree(Options);
      #endregion
   }
}
