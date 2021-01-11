using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionModelParameters;
using MSTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LightGbmRegressionTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal LightGbmRegressionTrainer(MachineLearningContext ml, MSTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.Regression.Trainers.LightGbm(Options);
      #endregion
   }
}
