using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di regressione
   /// </summary>
   public class RegressionTrainersCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal RegressionTrainersCatalog(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo FastForestRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastForestRegressionTrainer FastForest(Microsoft.ML.Trainers.FastTree.FastForestRegressionTrainer.Options options = default) =>
         new FastForestRegressionTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo FastTreeRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastTreeRegressionTrainer FastTree(Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Options options = default) =>
         new FastTreeRegressionTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo FastTreeRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastTreeTweedieTrainer FastTreeTweedie(Microsoft.ML.Trainers.FastTree.FastTreeTweedieTrainer.Options options = default) =>
         new FastTreeTweedieTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbmRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LightGbmRegressionTrainer LightGbm(Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options options = default) =>
         new LightGbmRegressionTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo SdcaRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaRegressionTrainer Sdca(Microsoft.ML.Trainers.SdcaRegressionTrainer.Options options = default) =>
         new SdcaRegressionTrainer(this, options);
      #endregion
   }
}
