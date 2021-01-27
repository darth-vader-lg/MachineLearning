using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Vision.ImageClassificationModelParameters;
using MSTrainer = Microsoft.ML.Vision.ImageClassificationTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class ImageClassificationTrainer :
      TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal ImageClassificationTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.ImageClassification(Options);
      #endregion
   }
}
