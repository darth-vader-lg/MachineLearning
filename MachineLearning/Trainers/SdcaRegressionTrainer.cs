using MachineLearning.Serialization;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Runtime.Serialization;
using ML = Microsoft.ML.Trainers;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaRegressionTrainer :
      TrainerBase<LinearRegressionModelParameters, ML.SdcaRegressionTrainer, ML.SdcaRegressionTrainer.Options>

   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal SdcaRegressionTrainer(MachineLearningContext ml, ML.SdcaRegressionTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override ML.SdcaRegressionTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.Regression.Trainers.Sdca(Options);
      #endregion
   }

   /// <summary>
   /// Surrogato delle opzioni per la serializzazione
   /// </summary>
   public partial class SdcaRegressionTrainer
   {
   }
}
