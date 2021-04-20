using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di regressione
   /// </summary>
   public class RegressionTrainers : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal RegressionTrainers(IContextProvider<MLContext> context) : base(context) { }
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
      /// Restituisce un trainer di tipo GamRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public GamRegressionTrainer Gam(Microsoft.ML.Trainers.FastTree.GamRegressionTrainer.Options options = default) =>
         new GamRegressionTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbmRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LbfgsPoissonRegressionTrainer LbfgsPoissonRegression(Microsoft.ML.Trainers.LbfgsPoissonRegressionTrainer.Options options = default) =>
         new LbfgsPoissonRegressionTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbmRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LightGbmRegressionTrainer LightGbm(Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options options = default) =>
         new LightGbmRegressionTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo Ols
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public OlsTrainer Ols(Microsoft.ML.Trainers.OlsTrainer.Options options = default) =>
         new OlsTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo OnlineGradientDescent
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public OnlineGradientDescentTrainer OnlineGradientDescentTrainer(Microsoft.ML.Trainers.OnlineGradientDescentTrainer.Options options = default) =>
         new OnlineGradientDescentTrainer(this, options);
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
