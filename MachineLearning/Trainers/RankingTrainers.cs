using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di ranking
   /// </summary>
   public class RankingTrainers : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal RankingTrainers(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo FastTreeRanking
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastTreeRankingTrainer FastTree(Microsoft.ML.Trainers.FastTree.FastTreeRankingTrainer.Options options = default) =>
         new FastTreeRankingTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbmRanking
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LightGbmRankingTrainer LightGbm(Microsoft.ML.Trainers.LightGbm.LightGbmRankingTrainer.Options options = default) =>
         new LightGbmRankingTrainer(this, options);
      #endregion
   }
}
