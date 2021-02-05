using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.MaximumEntropyModelParameters;
using MSTrainer = Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LbfgsMaximumEntropyMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class LbfgsMaximumEntropyMulticlassTrainer :
      TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal LbfgsMaximumEntropyMulticlassTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(Options);
      #endregion
   }
}
