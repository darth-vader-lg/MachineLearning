using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.MaximumEntropyModelParameters;
using MSTrainer = Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaMaximumEntropyMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaMaximumEntropyMulticlassTrainer :
      TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal SdcaMaximumEntropyMulticlassTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.SdcaMaximumEntropy(Options);
      #endregion
   }
}
