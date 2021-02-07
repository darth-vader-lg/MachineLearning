using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.MaximumEntropyModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaMaximumEntropyMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaMaximumEntropyMulticlassTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal SdcaMaximumEntropyMulticlassTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.SdcaMaximumEntropy(Options);
      #endregion
   }
}
