﻿using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di rilevamento anomalie
   /// </summary>
   public class AnomalyDetectionCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal AnomalyDetectionCatalog(IContextProvider<MLContext> context) : base(context) { }
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