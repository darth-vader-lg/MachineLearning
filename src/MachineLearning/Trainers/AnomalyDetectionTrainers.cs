using Microsoft.ML;
using System;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di rilevamento anomalie
   /// </summary>
   [Serializable]
   public class AnomalyDetectionTrainers : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal AnomalyDetectionTrainers(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo RandomizedPca
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public RandomizedPcaTrainer RandomizedPca(Microsoft.ML.Trainers.RandomizedPcaTrainer.Options options = default) =>
         new RandomizedPcaTrainer(this, options);
      #endregion
   }
}
