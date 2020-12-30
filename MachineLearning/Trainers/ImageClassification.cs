using Microsoft.ML;
using Microsoft.ML.Vision;
using System;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class ImageClassification :
      TrainerBase<ImageClassificationModelParameters, ImageClassificationTrainer, ImageClassificationTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml"></param>
      internal ImageClassification(MachineLearningContext ml, ImageClassificationTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override ImageClassificationTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.MulticlassClassification.Trainers.ImageClassification(Options);
      #endregion
   }
}
