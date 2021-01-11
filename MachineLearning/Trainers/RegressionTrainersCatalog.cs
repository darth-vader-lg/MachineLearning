namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di regressione
   /// </summary>
   public class RegressionTrainersCatalog
   {
      #region Fields
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      private readonly MachineLearningContext _ml;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal RegressionTrainersCatalog(MachineLearningContext ml) => _ml = ml;
      /// <summary>
      /// Restituisce un trainer di tipo SdcaRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LightGbmRegressionTrainer LightGbm(Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options options = default) =>
         new LightGbmRegressionTrainer(_ml, options);
      /// <summary>
      /// Restituisce un trainer di tipo SdcaRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaRegressionTrainer Sdca(Microsoft.ML.Trainers.SdcaRegressionTrainer.Options options = default) =>
         new SdcaRegressionTrainer(_ml, options);
      #endregion
   }
}
