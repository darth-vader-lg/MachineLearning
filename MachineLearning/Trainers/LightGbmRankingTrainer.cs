using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.LightGbm.LightGbmRankingTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmRankingTrainer;
using TTransformer = Microsoft.ML.Data.RankingPredictionTransformer<Microsoft.ML.Trainers.LightGbm.LightGbmRankingModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LightGbmRankingTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LightGbmRankingTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Ranking.Trainers.LightGbm(Options);
      #endregion
   }
}
