using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using ML = Microsoft.ML.Trainers;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaNonCalibratedMulticlassTrainer :
      TrainerBase<LinearMulticlassModelParameters, ML.SdcaNonCalibratedMulticlassTrainer, ML.SdcaNonCalibratedMulticlassTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal SdcaNonCalibratedMulticlassTrainer(MachineLearningContext ml, ML.SdcaNonCalibratedMulticlassTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override ML.SdcaNonCalibratedMulticlassTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.MulticlassClassification.Trainers.SdcaNonCalibrated(Options);
      #endregion
   }
}
