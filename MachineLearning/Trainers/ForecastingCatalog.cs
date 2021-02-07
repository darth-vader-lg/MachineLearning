using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di previsione
   /// </summary>
   public class ForecastingCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal ForecastingCatalog(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo ForecastBySsa
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public ForecastBySsaTrainer ForecastBySsa(ForecastBySsaTrainer.TrainerOptions options = default) =>
         new ForecastBySsaTrainer(this, options);
      #endregion
   }
}
