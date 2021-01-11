using Microsoft.ML;
using System;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SdcaRegressionTrainer :
      TrainerBase<Microsoft.ML.Trainers.LinearRegressionModelParameters,
         Microsoft.ML.Trainers.SdcaRegressionTrainer,
         Microsoft.ML.Trainers.SdcaRegressionTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal SdcaRegressionTrainer(MachineLearningContext ml, Microsoft.ML.Trainers.SdcaRegressionTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override Microsoft.ML.Trainers.SdcaRegressionTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.Regression.Trainers.Sdca(Options);
      #endregion
   }
}
