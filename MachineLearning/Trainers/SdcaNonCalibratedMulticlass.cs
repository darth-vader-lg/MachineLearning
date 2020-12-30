using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SdcaNonCalibratedMulticlass :
      TrainerBase<LinearMulticlassModelParameters, SdcaNonCalibratedMulticlassTrainer, SdcaNonCalibratedMulticlassTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal SdcaNonCalibratedMulticlass(MachineLearningContext ml, SdcaNonCalibratedMulticlassTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override SdcaNonCalibratedMulticlassTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.MulticlassClassification.Trainers.SdcaNonCalibrated(Options);
      #endregion
   }
}
