using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeTweedieModelParameters>;
using TTrainer = Microsoft.ML.Trainers.FastTree.FastTreeTweedieTrainer;
using TOptions = Microsoft.ML.Trainers.FastTree.FastTreeTweedieTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastTreeRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastTreeTweedieTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastTreeTweedieTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.FastTreeTweedie(Options);
      #endregion
   }
}
