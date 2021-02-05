using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.OneVersusAllModelParameters;
using MSTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class LightGbmMulticlassTrainer :
      TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal LightGbmMulticlassTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.LightGbm(Options);
      #endregion
   }
}
