using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using ML = Microsoft.ML.Vision;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class ImageClassificationTrainer :
      TrainerBase<ImageClassificationModelParameters, ML.ImageClassificationTrainer, ML.ImageClassificationTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml"></param>
      internal ImageClassificationTrainer(MachineLearningContext ml, ML.ImageClassificationTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override ML.ImageClassificationTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.MulticlassClassification.Trainers.ImageClassification(Options);
      #endregion
   }
}
