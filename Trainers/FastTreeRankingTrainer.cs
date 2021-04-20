using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.FastTree.FastTreeRankingTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.FastTree.FastTreeRankingTrainer;
using TTransformer = Microsoft.ML.Data.RankingPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastTreeRankingModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastTreeBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastTreeRankingTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastTreeRankingTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Ranking.Trainers.FastTree(Options);
      #endregion
   }
}
